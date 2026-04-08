/**
 * BE NOT AFRAID — Image Processing Pipeline
 * 
 * For each image in installation/images/:
 *   1. Claude Vision detects face bounds + eye regions + text regions
 *   2. rembg (Python) removes background → transparent PNG
 *   3. Eyes painted to transparent via detected coordinates
 *   4. Face mirrored bilaterally (left half → right) for angel effect
 *   5. Face enlarged to fill frame if too small
 *   6. Text regions receive extreme glitch corruption until unreadable
 *   7. Output saved to installation/processed/
 */

import Anthropic from '@anthropic-ai/sdk';
import sharp from 'sharp';
import { createCanvas, loadImage } from 'canvas';
import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'fs';
import { execSync } from 'child_process';
import { join, basename, extname } from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..', '..');
const INPUT_DIR = join(ROOT, 'installation', 'images');
const OUTPUT_DIR = join(ROOT, 'installation', 'processed');
const MANIFEST_PATH = join(OUTPUT_DIR, 'manifest.json');

if (!existsSync(OUTPUT_DIR)) mkdirSync(OUTPUT_DIR, { recursive: true });

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ───────────────────────────────────────────────────────────
// 1. CLAUDE VISION — detect faces, eyes, text
// ───────────────────────────────────────────────────────────
async function analyzeImage(imagePath) {
  const imageData = readFileSync(imagePath);
  const base64 = imageData.toString('base64');
  const ext = extname(imagePath).toLowerCase();
  const mediaType = ext === '.png' ? 'image/png' : 'image/jpeg';

  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 1000,
    messages: [{
      role: 'user',
      content: [
        {
          type: 'image',
          source: { type: 'base64', media_type: mediaType, data: base64 }
        },
        {
          type: 'text',
          text: `Analyze this image and return ONLY a JSON object with no markdown, no explanation:
{
  "imageWidth": <integer pixel width>,
  "imageHeight": <integer pixel height>,
  "face": {
    "found": <boolean>,
    "x": <left edge as 0-1 fraction of image width>,
    "y": <top edge as 0-1 fraction of image height>,
    "width": <face width as 0-1 fraction>,
    "height": <face height as 0-1 fraction>,
    "tooSmall": <true if face occupies less than 40% of image area>
  },
  "eyes": [
    { "cx": <center x 0-1>, "cy": <center y 0-1>, "r": <radius 0-1> },
    { "cx": <center x 0-1>, "cy": <center y 0-1>, "r": <radius 0-1> }
  ],
  "textRegions": [
    { "x": <0-1>, "y": <0-1>, "width": <0-1>, "height": <0-1> }
  ]
}
Be precise. eyes array may be empty if no eyes visible. textRegions may be empty.`
        }
      ]
    }]
  });

  const text = response.content.find(b => b.type === 'text')?.text || '{}';
  try {
    return JSON.parse(text.replace(/```json|```/g, '').trim());
  } catch {
    console.warn(`  Vision parse failed for ${basename(imagePath)}, using defaults`);
    return { face: { found: false }, eyes: [], textRegions: [] };
  }
}

// ───────────────────────────────────────────────────────────
// 2. BACKGROUND REMOVAL via rembg (Python)
// ───────────────────────────────────────────────────────────
function removeBackground(inputPath, outputPath) {
  execSync(`python3 -c "
from rembg import remove
from PIL import Image
import sys

with open('${inputPath}', 'rb') as f:
    data = f.read()

result = remove(data)

with open('${outputPath}', 'wb') as f:
    f.write(result)
print('rembg done')
"`, { stdio: 'inherit' });
}

// ───────────────────────────────────────────────────────────
// 3-6. SHARP/CANVAS POST-PROCESSING
//    - eye removal (transparent circles)
//    - bilateral face mirror
//    - face enlargement
//    - text glitch corruption
// ───────────────────────────────────────────────────────────
async function postProcess(pngPath, analysis, outputPath) {
  const img = await loadImage(pngPath);
  const W = img.width, H = img.height;

  const canvas = createCanvas(W, H);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);

  // ── Eye removal ──
  if (analysis.eyes && analysis.eyes.length) {
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    for (const eye of analysis.eyes) {
      const cx = eye.cx * W;
      const cy = eye.cy * H;
      // Use the larger of the two dimensions for radius — eyes should be fully gone
      const r = Math.max(eye.r * W, eye.r * H) * 1.3;
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
      grad.addColorStop(0.6, 'rgba(0,0,0,1)');
      grad.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(cx, cy, r, r * 0.7, 0, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  // ── Face enlargement if tooSmall ──
  let faceCanvas = canvas;
  if (analysis.face && analysis.face.found && analysis.face.tooSmall) {
    const fx = analysis.face.x * W;
    const fy = analysis.face.y * H;
    const fw = analysis.face.width * W;
    const fh = analysis.face.height * H;

    // Scale so face fills ~75% of frame
    const targetSize = Math.min(W, H) * 0.75;
    const scaleFactor = targetSize / Math.max(fw, fh);

    const enlargedCanvas = createCanvas(W, H);
    const eCtx = enlargedCanvas.getContext('2d');

    const scaledW = W * scaleFactor;
    const scaledH = H * scaleFactor;
    // Center the face
    const faceCX = fx + fw / 2;
    const faceCY = fy + fh / 2;
    const offsetX = W / 2 - faceCX * scaleFactor;
    const offsetY = H / 2 - faceCY * scaleFactor;

    eCtx.drawImage(canvas, offsetX, offsetY, scaledW, scaledH);
    faceCanvas = enlargedCanvas;
  }

  // ── Bilateral mirror — left half reflected to right ──
  // This is the "biblically accurate angel" effect
  const mirrorCanvas = createCanvas(W, H);
  const mCtx = mirrorCanvas.getContext('2d');

  // Draw left half
  mCtx.drawImage(faceCanvas, 0, 0);

  // Mirror left half onto right half
  mCtx.save();
  mCtx.scale(-1, 1);
  mCtx.drawImage(faceCanvas,
    // source: left half only
    0, 0, W / 2, H,
    // dest: right half (mirrored via scale)
    -W, 0, W / 2, H
  );
  mCtx.restore();

  // ── Text glitch corruption ──
  if (analysis.textRegions && analysis.textRegions.length) {
    const imageData = mCtx.getImageData(0, 0, W, H);
    const data = imageData.data;

    for (const region of analysis.textRegions) {
      const rx = Math.floor(region.x * W);
      const ry = Math.floor(region.y * H);
      const rw = Math.ceil(region.width * W);
      const rh = Math.ceil(region.height * H);

      // Pass 1: Horizontal scanline displacement
      for (let y = ry; y < ry + rh && y < H; y++) {
        const shift = Math.floor((Math.random() - 0.5) * rw * 0.8);
        for (let x = rx; x < rx + rw && x < W; x++) {
          const srcX = Math.max(0, Math.min(W - 1, x + shift));
          const srcIdx = (y * W + srcX) * 4;
          const dstIdx = (y * W + x) * 4;
          data[dstIdx]     = data[srcIdx];
          data[dstIdx + 1] = data[srcIdx + 1];
          data[dstIdx + 2] = data[srcIdx + 2];
        }
      }

      // Pass 2: Channel split — RGB offset
      for (let y = ry; y < ry + rh && y < H; y++) {
        for (let x = rx; x < rx + rw && x < W; x++) {
          const idx = (y * W + x) * 4;
          const rShiftX = Math.floor(Math.random() * 12 - 6);
          const gShiftX = Math.floor(Math.random() * 8 - 4);
          const rSrcIdx = (y * W + Math.max(0, Math.min(W-1, x + rShiftX))) * 4;
          const gSrcIdx = (y * W + Math.max(0, Math.min(W-1, x + gShiftX))) * 4;
          data[idx]     = data[rSrcIdx];
          data[idx + 1] = data[gSrcIdx + 1];
          // Blue channel stays — creates that classic chromatic aberration
        }
      }

      // Pass 3: Random block corruption
      const blockCount = 30 + Math.floor(Math.random() * 50);
      for (let b = 0; b < blockCount; b++) {
        const bx = rx + Math.floor(Math.random() * rw);
        const by = ry + Math.floor(Math.random() * rh);
        const bw = 2 + Math.floor(Math.random() * 20);
        const bh = 1 + Math.floor(Math.random() * 6);
        const val = Math.floor(Math.random() * 256);
        for (let py = by; py < by + bh && py < H; py++) {
          for (let px = bx; px < bx + bw && px < W; px++) {
            const i = (py * W + px) * 4;
            data[i]     = Math.random() > 0.5 ? val : 0;
            data[i + 1] = Math.random() > 0.5 ? val : 255;
            data[i + 2] = Math.floor(Math.random() * 256);
          }
        }
      }

      // Pass 4: Vertical smear
      for (let x = rx; x < rx + rw && x < W; x++) {
        if (Math.random() > 0.6) {
          const smearY = ry + Math.floor(Math.random() * rh);
          const smearLen = 3 + Math.floor(Math.random() * 20);
          const srcIdx = (smearY * W + x) * 4;
          for (let sy = smearY; sy < smearY + smearLen && sy < ry + rh && sy < H; sy++) {
            const dstIdx = (sy * W + x) * 4;
            data[dstIdx]     = data[srcIdx];
            data[dstIdx + 1] = data[srcIdx + 1];
            data[dstIdx + 2] = data[srcIdx + 2];
          }
        }
      }
    }

    mCtx.putImageData(imageData, 0, 0);
  }

  // ── Output ──
  const buffer = mirrorCanvas.toBuffer('image/png');
  writeFileSync(outputPath, buffer);
}

// ───────────────────────────────────────────────────────────
// MAIN
// ───────────────────────────────────────────────────────────
async function main() {
  console.log('═══ Be Not Afraid — Image Pipeline ═══\n');

  if (!existsSync(INPUT_DIR)) {
    console.log('No input directory found. Create installation/images/ and add images.');
    process.exit(0);
  }

  const validExts = ['.jpg', '.jpeg', '.png', '.webp'];
  const inputFiles = readdirSync(INPUT_DIR)
    .filter(f => validExts.includes(extname(f).toLowerCase()))
    .map(f => join(INPUT_DIR, f));

  if (!inputFiles.length) {
    console.log('No images found in installation/images/');
    process.exit(0);
  }

  console.log(`Found ${inputFiles.length} image(s) to process.\n`);

  const manifest = [];

  for (const inputPath of inputFiles) {
    const name = basename(inputPath, extname(inputPath));
    const outputPath = join(OUTPUT_DIR, `${name}.png`);

    console.log(`Processing: ${basename(inputPath)}`);

    try {
      // Step 1: Claude Vision analysis
      console.log('  → Claude Vision analysis...');
      const analysis = await analyzeImage(inputPath);
      console.log(`  → Face found: ${analysis.face?.found}, Eyes: ${analysis.eyes?.length || 0}, Text regions: ${analysis.textRegions?.length || 0}`);

      // Step 2: Background removal
      const bgRemovedPath = join(OUTPUT_DIR, `${name}_nobg.png`);
      console.log('  → Removing background...');
      removeBackground(inputPath, bgRemovedPath);

      // Step 3-6: Post-processing (eyes, mirror, enlarge, glitch)
      console.log('  → Mirroring, removing eyes, applying glitch...');
      await postProcess(bgRemovedPath, analysis, outputPath);

      // Cleanup temp file
      try { require('fs').unlinkSync(bgRemovedPath); } catch {}

      manifest.push({
        file: `${name}.png`,
        source: basename(inputPath),
        processedAt: new Date().toISOString(),
        hasFace: analysis.face?.found || false,
        eyeCount: analysis.eyes?.length || 0,
        textRegions: analysis.textRegions?.length || 0,
      });

      console.log(`  ✓ Saved: processed/${name}.png\n`);

    } catch (err) {
      console.error(`  ✗ Error processing ${basename(inputPath)}:`, err.message);
    }
  }

  // Write manifest for the frontend to reference
  writeFileSync(MANIFEST_PATH, JSON.stringify({
    generated: new Date().toISOString(),
    count: manifest.length,
    images: manifest,
  }, null, 2));

  console.log(`\n═══ Done: ${manifest.length}/${inputFiles.length} images processed ═══`);
}

main().catch(err => {
  console.error('Pipeline failed:', err);
  process.exit(1);
});
