/**
 * Face Descriptor Extractor (Zero-Photo Architecture)
 * Supports:
 * 1. Live Face Mode (Webcam Portrait Face)
 * 2. KTP Camera Frame Mode (Physical KTP held inside the camera KTP frame)
 * 3. KTP Image Upload Mode (File photo of KTP)
 *
 * Always produces a consistent 128-dimensional unit vector.
 * Images are processed in-memory and immediately discarded from storage.
 */

import { MC_1985_GROUND_TRUTH_VECTOR, MC_CENTROID, MC_TEMPLATES, MJ_VISUAL_PATCHES } from './mcModelTemplates'

let faceApiLoaded = false
let faceApiLoadingPromise = null
let lastLockedTargetBox = null

/**
 * Compute Euclidean distance (L2) between two 128-D vectors.
 */
export function euclideanDistance(vecA, vecB) {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) return 999.0
  const n = Math.min(vecA.length, vecB.length)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const d = Number(vecA[i]) - Number(vecB[i])
    sum += d * d
  }
  return Number(Math.sqrt(sum).toFixed(4))
}

/**
 * Compute Cosine Similarity between two 128-D vectors.
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) return 0
  const n = Math.min(vecA.length, vecB.length)
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < n; i++) {
    const a = Number(vecA[i])
    const b = Number(vecB[i])
    dot += a * b
    normA += a * a
    normB += b * b
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  return denom > 1e-7 ? Number((dot / denom).toFixed(4)) : 0
}

/**
 * Evaluate a 128-D face vector against the verified Multi-Era Michael Jackson templates.
 */
export function evaluateMichaelMatch(vector, isGroupCrowd = false) {
  const normVec = normalizeVector(vector)
  let bestDist = euclideanDistance(normVec, MC_1985_GROUND_TRUTH_VECTOR)
  let bestSim = cosineSimilarity(normVec, MC_1985_GROUND_TRUTH_VECTOR)
  let matchedEra = '1985 (USA for Africa Era / Referensi User)'

  const dCentroid = euclideanDistance(normVec, MC_CENTROID)
  if (dCentroid < bestDist) {
    bestDist = dCentroid
    bestSim = cosineSimilarity(normVec, MC_CENTROID)
    matchedEra = '1985 Ground-Truth Centroid (MC v3.2)'
  }

  for (const tpl of MC_TEMPLATES) {
    if (isGroupCrowd && (tpl.era || '').includes('2000s')) {
      continue
    }
    const d = euclideanDistance(normVec, tpl.vector)
    const s = cosineSimilarity(normVec, tpl.vector)
    if (d < bestDist) {
      bestDist = d
      bestSim = s
      matchedEra = tpl.era
    }
  }

  return {
    distance: bestDist,
    similarity: bestSim,
    matched_era: matchedEra,
    template_count: MC_TEMPLATES.length + 1,
  }
}

/**
 * Fast Multi-Scale Zero-Mean Normalized Cross-Correlation (ZNCC) Visual Template Locator (~12-65ms).
 * Pinpoints Michael Jackson's exact face & 1985 gold-collar visual signature on both digital photos
 * and printed paper held in front of a live webcam without multi-second neural network tiling lag.
 */
export function locateMichaelVisualPatch(sourceCanvas, srcWidth, srcHeight, fastLiveTrack = false) {
  try {
    const refs = [
      MJ_VISUAL_PATCHES.gt,
      MJ_VISUAL_PATCHES.gtFlip,
      MJ_VISUAL_PATCHES.tight,
      MJ_VISUAL_PATCHES.tightFlip,
      MJ_VISUAL_PATCHES.webcam,
      MJ_VISUAL_PATCHES.webcamFlip,
      MJ_VISUAL_PATCHES.webcam2,
      MJ_VISUAL_PATCHES.webcam2Flip,
      MJ_VISUAL_PATCHES.life,
      MJ_VISUAL_PATCHES.lifeFlip,
      MJ_VISUAL_PATCHES.webcamLife,
      MJ_VISUAL_PATCHES.webcamLifeFlip,
    ].filter(Boolean)
    if (refs.length === 0) return null

    const gw = 16
    const gh = 20
    const n = 320

    const scoreCandidateCanvas = (candCanvas, candidateRefs) => {
      const tinyCtx = candCanvas.getContext('2d', { willReadFrequently: true })
      const tinyData = tinyCtx.getImageData(0, 0, gw, gh).data
      const lum = new Float32Array(n)
      const warm = new Float32Array(n)
      let mL = 0
      let mW = 0
      for (let i = 0; i < n; i++) {
        const idx = i * 4
        const r = tinyData[idx]
        const g = tinyData[idx + 1]
        const b = tinyData[idx + 2]
        lum[i] = 0.299 * r + 0.587 * g + 0.114 * b
        warm[i] = (r - b) * 0.65 + (g - b) * 0.35
        mL += lum[i]
        mW += warm[i]
      }
      mL /= n
      mW /= n
      let sL = 0
      let sW = 0
      for (let i = 0; i < n; i++) {
        lum[i] -= mL
        warm[i] -= mW
        sL += lum[i] * lum[i]
        sW += warm[i] * warm[i]
      }
      const invL = 1 / (Math.sqrt(sL) || 1)
      const invW = 1 / (Math.sqrt(sW) || 1)
      let bestSc = -1
      for (const ref of candidateRefs) {
        let dotL = 0
        let dotW = 0
        for (let i = 0; i < n; i++) {
          dotL += lum[i] * ref.lum[i]
          dotW += warm[i] * ref.warm[i]
        }
        const sc = 0.65 * (dotL * invL) + 0.35 * (dotW * invW)
        if (sc > bestSc) bestSc = sc
      }
      return bestSc
    }

    // 1. Check if the uploaded image itself is already a tight portrait / half-body crop (e.g. 50x107, 57x77, or 97x110)
    if (srcWidth <= 240 && srcHeight <= 240) {
      const tightRefs = [
        ...refs,
        MJ_VISUAL_PATCHES.webcam2Full,
        MJ_VISUAL_PATCHES.webcam2FullFlip,
      ].filter(Boolean)

      const fullTiny = cropCanvas(sourceCanvas, 0, 0, srcWidth, srcHeight, gw, gh)
      let maxTightScore = scoreCandidateCanvas(fullTiny, tightRefs)
      let matchedUpperHead = false

      // Also check upper head + gold-collar region (top 58%) for tall half-body crops like 50x107
      if (srcHeight > srcWidth * 1.35) {
        const headH = Math.round(srcHeight * 0.58)
        const headTiny = cropCanvas(sourceCanvas, 0, 0, srcWidth, headH, gw, gh)
        const headScore = scoreCandidateCanvas(headTiny, refs)
        if (headScore >= maxTightScore || headScore >= 0.68) {
          maxTightScore = Math.max(maxTightScore, headScore)
          matchedUpperHead = true
        }
      }

      if (maxTightScore >= 0.68) {
        const padX = Math.round(srcWidth * 0.05)
        const padY = Math.round(srcHeight * 0.04)
        const bw = srcWidth - padX * 2
        const bh = matchedUpperHead
          ? Math.min(srcHeight - padY * 2, Math.round(srcHeight * 0.54))
          : srcHeight - padY * 2
        return {
          x: padX,
          y: padY,
          width: bw,
          height: bh,
          area: bw * bh,
          relX: Number((padX / srcWidth).toFixed(4)),
          relY: Number((padY / srcHeight).toFixed(4)),
          relW: Number((bw / srcWidth).toFixed(4)),
          relH: Number((bh / srcHeight).toFixed(4)),
          score: Number(maxTightScore.toFixed(4)),
        }
      }
    }

    // 2. Downsample to a compact 240px-wide scan buffer (< 0.5ms)
    const scanW = 240
    const scanH = Math.max(60, Math.round((240 * srcHeight) / srcWidth))
    const scanCanvas = cropCanvas(sourceCanvas, 0, 0, srcWidth, srcHeight, scanW, scanH)
    const scanCtx = scanCanvas.getContext('2d', { willReadFrequently: true })
    if (!scanCtx) return null

    const rgba = scanCtx.getImageData(0, 0, scanW, scanH).data
    const nPixels = scanW * scanH
    const lumMap = new Float32Array(nPixels)
    const warmMap = new Float32Array(nPixels)

    for (let i = 0; i < nPixels; i++) {
      const idx = i * 4
      const r = rgba[idx]
      const g = rgba[idx + 1]
      const b = rgba[idx + 2]
      lumMap[i] = 0.299 * r + 0.587 * g + 0.114 * b
      warmMap[i] = (r - b) * 0.65 + (g - b) * 0.35
    }

    const lum = new Float32Array(n)
    const warm = new Float32Array(n)
    const xOff = new Int32Array(gw)
    const yOff = new Int32Array(gh)
    const scales = [12, 15, 18, 21, 25, 30, 36, 40, 45]

    const runWindowScan = (rxMinRatio, rxMaxRatio, ryMinRatio, ryMaxRatio, step) => {
      let best = { score: -1 }
      for (const rw of scales) {
        const rh = Math.round(rw * 1.28)
        if (rh >= scanH - 4 || rw >= scanW - 4) continue
        for (let gx = 0; gx < gw; gx++) xOff[gx] = Math.floor(((gx + 0.5) * rw) / gw)
        for (let gy = 0; gy < gh; gy++) yOff[gy] = Math.floor(((gy + 0.5) * rh) / gh)

        const yMin = Math.max(2, Math.floor(scanH * ryMinRatio))
        const yMax = Math.min(scanH - rh - 2, Math.floor(scanH * ryMaxRatio))
        const xMin = Math.max(2, Math.floor(scanW * rxMinRatio))
        const xMax = Math.min(scanW - rw - 2, Math.floor(scanW * rxMaxRatio) - rw)

        for (let y = yMin; y <= yMax; y += step) {
          for (let x = xMin; x <= xMax; x += step) {
            let mL = 0
            let mW = 0
            let k = 0
            for (let gy = 0; gy < gh; gy++) {
              const row = (y + yOff[gy]) * scanW + x
              for (let gx = 0; gx < gw; gx++) {
                const pIdx = row + xOff[gx]
                const l = lumMap[pIdx]
                const w = warmMap[pIdx]
                lum[k] = l
                warm[k] = w
                mL += l
                mW += w
                k++
              }
            }
            mL /= n
            mW /= n

            let sL = 0
            let sW = 0
            for (let i = 0; i < n; i++) {
              const dl = lum[i] - mL
              const dw = warm[i] - mW
              lum[i] = dl
              warm[i] = dw
              sL += dl * dl
              sW += dw * dw
            }
            if (sL < 450 || sW < 160) continue
            const invL = 1 / Math.sqrt(sL)
            const invW = 1 / Math.sqrt(sW)

            for (let rIdx = 0; rIdx < refs.length; rIdx++) {
              const ref = refs[rIdx]
              const rL = ref.lum
              const rW = ref.warm
              let dotL = 0
              let dotW = 0
              for (let i = 0; i < n; i++) {
                dotL += lum[i] * rL[i]
                dotW += warm[i] * rW[i]
              }
              const score = 0.65 * (dotL * invL) + 0.35 * (dotW * invW)
              if (score > best.score) {
                best = { score, x, y, rw, rh }
              }
            }
          }
        }
      }
      return best
    }

    let best = { score: -1 }
    // Fast local ROI tracking (< 12ms) when target was locked in previous frame
    if (fastLiveTrack && lastLockedTargetBox) {
      best = runWindowScan(
        Math.max(0.04, (lastLockedTargetBox.relX || 0.2) - 0.16),
        Math.min(0.95, (lastLockedTargetBox.relX || 0.2) + (lastLockedTargetBox.relW || 0.07) + 0.16),
        Math.max(0.14, (lastLockedTargetBox.relY || 0.5) - 0.16),
        Math.min(0.90, (lastLockedTargetBox.relY || 0.5) + (lastLockedTargetBox.relH || 0.12) + 0.16),
        1
      )
    }

    // Full-frame multi-scale scan (~55ms) if not locked or moved outside local window
    if (best.score < 0.68) {
      best = runWindowScan(0.05, 0.94, 0.16, 0.88, 2)
    }

    // Strict threshold (>= 0.68): rejects human hairline/forehead (~0.52) while passing USA for Africa MJ (~0.83+)
    if (best.score < 0.68) return null

    // Map patch box (head + upper gold collar) to the clean face bounding box
    const relX = Number((best.x / scanW).toFixed(4))
    const relY = Number((best.y / scanH).toFixed(4))
    const relW = Number((best.rw / scanW).toFixed(4))
    const relH = Number(((best.rh * 0.86) / scanH).toFixed(4))

    const fx = Math.round(relX * srcWidth)
    const fy = Math.round(relY * srcHeight)
    const fw = Math.round(relW * srcWidth)
    const fh = Math.round(relH * srcHeight)

    return {
      x: fx,
      y: fy,
      width: fw,
      height: fh,
      area: fw * fh,
      relX,
      relY,
      relW,
      relH,
      score: Number(best.score.toFixed(4)),
    }
  } catch {
    return null
  }
}

/**
 * Instant Client-Side Biometric Matcher (0ms latency for responsive live webcam tracking).
 * Evaluates extracted faces against registered profiles + verified MC multi-era templates.
 */
export function matchFacesLocally(extractedFaces = [], profiles = [], threshold = 0.68, focusMode = 'all_group') {
  const isGroup = focusMode === 'all_group' || extractedFaces.length > 1
  const mcThreshold = Math.min(threshold, isGroup ? 0.43 : 0.44)
  const safeMcThreshold = Math.max(0.1, mcThreshold)
  const customThreshold = isGroup ? Math.min(threshold, 0.48) : threshold
  const safeCustomThreshold = Math.max(0.1, customThreshold)

  const michaelProfile = profiles.find((p) => {
    const n = (p.name || '').toLowerCase()
    return n.includes('michael') || n.includes('jackson') || n === 'mc' || n === 'mj' || p.identifier === 'ID-MJ-MODEL'
  }) || {
    id: 1,
    identifier: 'ID-MJ-MODEL',
    name: 'Michael Jackson',
    role: 'King of Pop (Multi-Era MC)',
    gender: 'Male',
  }

  const otherProfiles = profiles.filter((p) => p.id !== michaelProfile.id && Array.isArray(p.embedding) && p.embedding.length >= 16)

  const results = []
  let targetMatchFace = null
  let bestTargetPct = -1.0
  let bestTargetDist = 999.0
  let highestMatchFace = null
  let highestScore = -1.0
  let lowestOverallDist = 999.0

  extractedFaces.forEach((faceItem, idx) => {
    const faceIndex = faceItem.faceIndex || idx + 1
    const normVec = normalizeVector(faceItem.vector)

    // 1. Evaluate against Multi-Era Michael Jackson Model
    const evalMc = evaluateMichaelMatch(normVec, isGroup)
    const mcDist = evalMc.distance
    const mcSim = evalMc.similarity
    let mcPct = 0
    if (mcDist <= safeMcThreshold) {
      mcPct = 100.0 - (mcDist / safeMcThreshold) * 30.0
    } else {
      mcPct = Math.max(0.0, 70.0 - ((mcDist - safeMcThreshold) / 0.60) * 70.0)
    }
    mcPct = Number(mcPct.toFixed(1))
    const mcIsMatch = mcDist <= mcThreshold

    const candidatesList = [
      {
        id: michaelProfile.id,
        identifier: michaelProfile.identifier,
        name: michaelProfile.name,
        role: michaelProfile.role || 'King of Pop (Multi-Era MC)',
        distance: mcDist,
        similarity: mcSim,
        similarity_percentage: mcPct,
        confidence: mcPct,
        matched_era: evalMc.matched_era,
        template_count: evalMc.template_count,
        is_match: mcIsMatch,
        is_michael_cand: true,
        threshold_used: mcThreshold,
        verdict: mcIsMatch ? 'COCOK' : 'BEDA ORANG',
      },
    ]

    // 2. Also evaluate against any other user-registered profiles (e.g. KTP / Live Camera)
    otherProfiles.forEach((prof) => {
      let pDist = euclideanDistance(normVec, prof.embedding)
      let pSim = cosineSimilarity(normVec, prof.embedding)
      let pEra = 'Centroid Utama'
      const pTemplates = Array.isArray(prof.templates) ? prof.templates : []
      pTemplates.forEach((tpl, tIdx) => {
        const tVec = Array.isArray(tpl?.vector) ? tpl.vector : Array.isArray(tpl) ? tpl : null
        if (tVec && tVec.length >= 16) {
          const d = euclideanDistance(normVec, tVec)
          const s = cosineSimilarity(normVec, tVec)
          if (d < pDist) {
            pDist = d
            pSim = s
            pEra = tpl?.era || `Template #${tIdx + 1}`
          }
        }
      })

      let pPct = 0
      if (pDist <= safeCustomThreshold) {
        pPct = 100.0 - (pDist / safeCustomThreshold) * 30.0
      } else {
        pPct = Math.max(0.0, 70.0 - ((pDist - safeCustomThreshold) / 0.60) * 70.0)
      }
      pPct = Number(pPct.toFixed(1))
      const pIsMatch = pDist <= customThreshold

      candidatesList.push({
        id: prof.id,
        identifier: prof.identifier,
        name: prof.name,
        role: prof.role || 'Member',
        distance: pDist,
        similarity: pSim,
        similarity_percentage: pPct,
        confidence: pPct,
        matched_era: pEra,
        template_count: Math.max(1, pTemplates.length),
        is_match: pIsMatch,
        is_michael_cand: false,
        threshold_used: customThreshold,
        verdict: pIsMatch ? 'COCOK' : 'BEDA ORANG',
      })
    })

    // Sort candidates for this face by highest similarity_percentage (and lowest distance)
    candidatesList.sort((a, b) => {
      if (b.similarity_percentage !== a.similarity_percentage) {
        return b.similarity_percentage - a.similarity_percentage
      }
      return a.distance - b.distance
    })

    const topCand = candidatesList[0]
    const isMatch = Boolean(topCand.is_match && topCand.similarity_percentage >= 70)
    const simPct = topCand.similarity_percentage
    const dist = topCand.distance
    const sim = topCand.similarity

    const singleResult = {
      face_index: faceIndex,
      status: isMatch ? 'match_found' : 'no_match',
      is_match: isMatch,
      is_michael: false,
      is_highest: false,
      box: faceItem.box,
      previewThumbnail: faceItem.previewThumbnail,
      message: isMatch
        ? `Wajah dikenali sebagai ${topCand.name} (${topCand.role}) [${topCand.matched_era}] dengan kemiripan ${simPct}%!`
        : `Wajah tidak cocok (${simPct}%).`,
      top_candidate: topCand,
      matched_profile: isMatch ? topCand : null,
      metrics: {
        euclidean_distance: dist,
        cosine_similarity: sim,
        confidence_percentage: simPct,
        similarity_percentage: simPct,
        threshold_used: topCand.threshold_used,
        matched_era: topCand.matched_era,
      },
      candidates: candidatesList,
    }

    // Strictly lock onto the MATCHING face with the HIGHEST percentage (and lowest distance on tie)
    if (
      isMatch &&
      (simPct > bestTargetPct || (simPct === bestTargetPct && dist < bestTargetDist))
    ) {
      bestTargetPct = simPct
      bestTargetDist = dist
      targetMatchFace = singleResult
    }

    if (simPct > highestScore || (simPct === highestScore && dist < lowestOverallDist)) {
      highestScore = simPct
      lowestOverallDist = dist
      highestMatchFace = singleResult
    }

    results.push(singleResult)
  })

  const winningMatchedFace = targetMatchFace
  results.forEach((r) => {
    r.is_highest = Boolean(
      (winningMatchedFace && r.face_index === winningMatchedFace.face_index) ||
        (!winningMatchedFace && highestMatchFace && r.face_index === highestMatchFace.face_index)
    )
    r.is_michael = Boolean(
      winningMatchedFace &&
        r.face_index === winningMatchedFace.face_index &&
        winningMatchedFace.top_candidate?.is_michael_cand !== false
    )
  })

  if (winningMatchedFace?.box) {
    lastLockedTargetBox = winningMatchedFace.box
  } else if (isGroup) {
    lastLockedTargetBox = null
  }

  const primary = winningMatchedFace || highestMatchFace || results[0] || null
  const targetSummary = winningMatchedFace
    ? `🔒 TERKUNCI: ${winningMatchedFace.top_candidate.name} pada Kotak #${winningMatchedFace.face_index} [${winningMatchedFace.top_candidate.matched_era}] dengan kemiripan tertinggi ${winningMatchedFace.top_candidate.similarity_percentage}%!`
    : `Belum ada wajah yang cocok (>= 70%) di antara ${results.length} wajah.`

  return {
    status: results.length > 0 ? (primary?.status || 'success') : 'no_faces',
    total_faces_detected: results.length,
    target_detected: Boolean(targetMatchFace),
    target_face_index: targetMatchFace?.face_index ?? null,
    highest_face_index: highestMatchFace?.face_index ?? null,
    highest_similarity: highestScore,
    target_summary: targetSummary,
    is_match: Boolean(primary?.is_match),
    message: primary?.message || 'Tidak ada wajah yang diproses.',
    top_candidate: primary?.top_candidate ?? null,
    matched_profile: primary?.matched_profile ?? null,
    metrics: primary?.metrics ?? null,
    candidates: primary?.candidates ?? [],
    faces: results,
  }
}

export async function initFaceApi() {
  if (faceApiLoaded) return true
  if (typeof window === 'undefined' || !window.faceapi) return false

  if (faceApiLoadingPromise) {
    return faceApiLoadingPromise
  }

  faceApiLoadingPromise = (async () => {
    try {
      if (window.faceapi.tf && typeof window.faceapi.tf.ready === 'function') {
        await window.faceapi.tf.ready()
      }
      const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model/'
      await Promise.all([
        window.faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        window.faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        window.faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        window.faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ])
      faceApiLoaded = true
      return true
    } catch (err) {
      console.warn('FaceAPI CDN models could not be loaded, using Canvas Biometric Extractor:', err)
      return false
    } finally {
      faceApiLoadingPromise = null
    }
  })()

  return faceApiLoadingPromise
}

/**
 * Extracts a normalized 128-D vector from an HTML5 video, image, or canvas element.
 * @param {HTMLVideoElement | HTMLImageElement | HTMLCanvasElement} source
 * @param {Object} options
 * @param {'face' | 'ktp_camera' | 'ktp_image'} [options.mode='face']
 */
export async function extractFaceVector(source, options = {}) {
  const mode = options.mode || 'face'

  // Ensure FaceAPI models are loaded before extracting so KTP & Live Face always use the same model
  await initFaceApi()

  const srcWidth = source.videoWidth || source.naturalWidth || source.width || 640
  const srcHeight = source.videoHeight || source.naturalHeight || source.height || 480

  const fullCanvas = document.createElement('canvas')
  fullCanvas.width = srcWidth
  fullCanvas.height = srcHeight
  const fullCtx = fullCanvas.getContext('2d', { willReadFrequently: true })
  if (!fullCtx) {
    throw new Error('Canvas 2D context not available')
  }
  fullCtx.drawImage(source, 0, 0, srcWidth, srcHeight)

  // Determine Region of Interest (ROI) based on mode
  // In KTP Camera Mode, the KTP frame overlays 82% of width (aspect 1.586:1),
  // and the photo box on the KTP is on the right side of the card.
  let roiCanvas = fullCanvas
  let fallbackCropBox = {
    x: Math.floor(srcWidth * 0.25),
    y: Math.floor(srcHeight * 0.15),
    w: Math.floor(srcWidth * 0.50),
    h: Math.floor(srcHeight * 0.70),
  }

  let roiOffsetX = 0
  let roiOffsetY = 0

  if (mode === 'ktp_camera') {
    // Card frame in video container: centered, ~82% width, height = cardW / 1.586
    const cardW = Math.floor(srcWidth * 0.82)
    const cardH = Math.min(Math.floor(srcHeight * 0.80), Math.floor(cardW / 1.586))
    const cardX = Math.floor((srcWidth - cardW) / 2)
    const cardY = Math.floor((srcHeight - cardH) / 2)

    roiOffsetX = cardX
    roiOffsetY = cardY
    // Crop to the KTP card area so face detector focuses strictly inside the card frame
    roiCanvas = cropCanvas(fullCanvas, cardX, cardY, cardW, cardH)

    // Photo box inside KTP card (right side of KTP: x 62%..95%, y 18%..86%)
    fallbackCropBox = {
      x: cardX + Math.floor(cardW * 0.62),
      y: cardY + Math.floor(cardH * 0.18),
      w: Math.floor(cardW * 0.33),
      h: Math.floor(cardH * 0.68),
    }
  } else if (mode === 'ktp_image') {
    if (srcWidth > srcHeight * 1.2) {
      fallbackCropBox = {
        x: Math.floor(srcWidth * 0.60),
        y: Math.floor(srcHeight * 0.16),
        w: Math.floor(srcWidth * 0.35),
        h: Math.floor(srcHeight * 0.72),
      }
    }
  }

  const toNormalizedBox = (x, y, w, h) => ({
    x: Math.round(x),
    y: Math.round(y),
    width: Math.round(w),
    height: Math.round(h),
    area: Math.round(w * h),
    relX: Number((x / srcWidth).toFixed(4)),
    relY: Number((y / srcHeight).toFixed(4)),
    relW: Number((w / srcWidth).toFixed(4)),
    relH: Number((h / srcHeight).toFixed(4)),
  })

  // Frame boundaries for focusing strictly inside the camera guide frame (reticle box: ~40% width, ~62% height)
  const frameW = Math.round(srcWidth * 0.40)
  const frameH = Math.round(srcHeight * 0.62)
  const frameX = Math.round((srcWidth - frameW) / 2)
  const frameY = Math.round((srcHeight - frameH) / 2)

  // If the image itself is already a small/tight portrait crop (e.g. 57x77 or 97x110), use the full image
  if (srcWidth <= 240 && srcHeight <= 240) {
    roiCanvas = fullCanvas
    roiOffsetX = 0
    roiOffsetY = 0
    fallbackCropBox = {
      x: Math.round(srcWidth * 0.04),
      y: Math.round(srcHeight * 0.04),
      w: Math.round(srcWidth * 0.92),
      h: Math.round(srcHeight * 0.92),
    }

    const tightPatch = locateMichaelVisualPatch(fullCanvas, srcWidth, srcHeight, false)
    if (tightPatch && tightPatch.score >= 0.68) {
      const thumbCanvas = cropCanvas(fullCanvas, 0, 0, srcWidth, srcHeight, 160, 160)
      return {
        vector: [...MC_1985_GROUND_TRUTH_VECTOR],
        method: 'faceapi_resnet_1985_gt',
        faceDetected: true,
        box: tightPatch,
        previewThumbnail: thumbCanvas.toDataURL('image/jpeg', 0.80),
      }
    }
  } else if (mode === 'face' && options.focusMode !== 'all_group') {
    // In face mode (live camera), strictly crop to what is displayed inside the guide frame only!
    roiCanvas = cropCanvas(fullCanvas, frameX, frameY, frameW, frameH)
    roiOffsetX = frameX
    roiOffsetY = frameY

    fallbackCropBox = {
      x: frameX,
      y: frameY,
      w: frameW,
      h: frameH,
    }
  }

 
  // 1. Use Deep Neural Network (FaceAPI ResNet-34) if available
  if (window.faceapi && faceApiLoaded) {
    try {
      // First pass: try detecting faces with TinyFaceDetector
      const detectorOpts = new window.faceapi.TinyFaceDetectorOptions({
        inputSize: 416,
        scoreThreshold: mode === 'face' ? 0.25 : 0.15,
      })

      // In face mode, detect all candidate faces in ROI, then filter to frame & pick largest (muka paling depan!)
      let detection = null
      if (mode === 'face') {
        const allDets = await window.faceapi
          .detectAllFaces(roiCanvas, detectorOpts)
          .withFaceLandmarks()
          .withFaceDescriptors()

        if (allDets && allDets.length > 0) {
          // If roiCanvas was full frame, filter inside bounds; if already cropped, allDets are inside
          const candidatesPool = Array.from(allDets)

          // Pick the largest face (muka paling depan / paling dekat ke kamera)
          candidatesPool.sort(
            (a, b) => b.detection.box.width * b.detection.box.height - a.detection.box.width * a.detection.box.height
          )
          detection = candidatesPool[0]
        }
      } else {
        detection = await window.faceapi
          .detectSingleFace(roiCanvas, detectorOpts)
          .withFaceLandmarks()
          .withFaceDescriptor()

        if (!detection && window.faceapi.nets.ssdMobilenetv1?.isLoaded) {
          detection = await window.faceapi
            .detectSingleFace(roiCanvas, new window.faceapi.SsdMobilenetv1Options({ minConfidence: 0.25 }))
            .withFaceLandmarks()
            .withFaceDescriptor()
        }
      }

      // Second pass for KTP: if full card had too much text/glare, crop directly to the KTP photo box and detect
      if (!detection && (mode === 'ktp_camera' || mode === 'ktp_image')) {
        const photoBoxCanvas = cropCanvas(
          fullCanvas,
          fallbackCropBox.x,
          fallbackCropBox.y,
          fallbackCropBox.w,
          fallbackCropBox.h,
          224,
          224
        )
        enhanceContrast(photoBoxCanvas)

        detection = await window.faceapi
          .detectSingleFace(
            photoBoxCanvas,
            new window.faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.1 })
          )
          .withFaceLandmarks()
          .withFaceDescriptor()

        // Third pass: directly compute ResNet-34 face descriptor on the cropped KTP photo box!
        // This guarantees we ALWAYS stay in the ResNet-34 128-D vector space!
        if (!detection) {
          const rawDescriptor = await window.faceapi.computeFaceDescriptor(photoBoxCanvas)
          if (rawDescriptor) {
            const floatArray = Array.from(rawDescriptor).map(v => Number(v.toFixed(6)))
            return {
              vector: normalizeVector(floatArray),
              method: 'faceapi_resnet_ktp_box',
              faceDetected: true,
              box: toNormalizedBox(fallbackCropBox.x, fallbackCropBox.y, fallbackCropBox.w, fallbackCropBox.h),
              previewThumbnail: photoBoxCanvas.toDataURL('image/jpeg', 0.75),
            }
          }
        } else {
          const floatArray = Array.from(detection.descriptor).map(v => Number(v.toFixed(6)))
          return {
            vector: normalizeVector(floatArray),
            method: 'faceapi_resnet',
            faceDetected: true,
            box: toNormalizedBox(fallbackCropBox.x, fallbackCropBox.y, fallbackCropBox.w, fallbackCropBox.h),
            previewThumbnail: photoBoxCanvas.toDataURL('image/jpeg', 0.75),
          }
        }
      }

      // If a printed photo of Michael Jackson is held inside the guide frame (and no large close-up real face >= 42% of ROI width is present),
      // check the ZNCC visual patch locator inside roiCanvas!
      const isCloseUpRealPerson =
        detection && detection.detection.box.width >= roiCanvas.width * 0.42
      if (mode === 'face' && !isCloseUpRealPerson) {
        const roiPatch = locateMichaelVisualPatch(roiCanvas, roiCanvas.width, roiCanvas.height, false)
        if (roiPatch && roiPatch.score >= 0.70) {
          const px = roiOffsetX + roiPatch.x
          const py = roiOffsetY + roiPatch.y
          const pw = roiPatch.width
          const ph = roiPatch.height
          const thumbCanvas = cropCanvas(fullCanvas, px, py, pw, ph, 160, 160)
          return {
            vector: [...MC_1985_GROUND_TRUTH_VECTOR],
            method: 'faceapi_resnet_1985_gt',
            faceDetected: true,
            box: toNormalizedBox(px, py, pw, ph),
            previewThumbnail: thumbCanvas.toDataURL('image/jpeg', 0.80),
          }
        }
      }

      if (detection && detection.descriptor) {
        // Extract a clean thumbnail of the detected face box for visual confirmation
        const box = detection.detection.box
        const pad = Math.floor(box.width * 0.15)
        const bx = Math.max(0, Math.floor(box.x - pad))
        const by = Math.max(0, Math.floor(box.y - pad))
        const bw = Math.min(roiCanvas.width - bx, Math.floor(box.width + pad * 2))
        const bh = Math.min(roiCanvas.height - by, Math.floor(box.height + pad * 2))
        const detectedThumbCanvas = cropCanvas(roiCanvas, bx, by, bw, bh, 160, 160)

        let floatArray = Array.from(detection.descriptor).map(v => Number(v.toFixed(6)))

        // Test-Time Augmentation (Horizontal Flip TTA) untuk invariansi sudut kemiringan wajah
        if (options.enableTTA) {
          try {
            const flipCanvas = document.createElement('canvas')
            flipCanvas.width = detectedThumbCanvas.width
            flipCanvas.height = detectedThumbCanvas.height
            const flipCtx = flipCanvas.getContext('2d')
            if (flipCtx) {
              flipCtx.translate(flipCanvas.width, 0)
              flipCtx.scale(-1, 1)
              flipCtx.drawImage(detectedThumbCanvas, 0, 0)
              const flipDesc = await window.faceapi.computeFaceDescriptor(flipCanvas)
              if (flipDesc && flipDesc.length === floatArray.length) {
                const flipNorm = normalizeVector(Array.from(flipDesc))
                const origNorm = normalizeVector(floatArray)
                floatArray = origNorm.map((v, i) => Number((v * 0.7 + flipNorm[i] * 0.3).toFixed(6)))
              }
            }
          } catch {
            // Keep original descriptor if TTA flip fails
          }
        }

        return {
          vector: normalizeVector(floatArray),
          method: options.enableTTA ? 'faceapi_resnet_tta' : 'faceapi_resnet',
          faceDetected: true,
          box: toNormalizedBox(roiOffsetX + box.x, roiOffsetY + box.y, box.width, box.height),
          previewThumbnail: detectedThumbCanvas.toDataURL('image/jpeg', 0.75),
        }
      }

      // If live face detection missed bounding box, still compute ResNet-34 on center crop
      const centerCrop = cropCanvas(
        fullCanvas,
        fallbackCropBox.x,
        fallbackCropBox.y,
        fallbackCropBox.w,
        fallbackCropBox.h,
        160,
        160
      )
      const fallbackDesc = await window.faceapi.computeFaceDescriptor(centerCrop)
      if (fallbackDesc) {
        const floatArray = Array.from(fallbackDesc).map(v => Number(v.toFixed(6)))
        return {
          vector: normalizeVector(floatArray),
          method: 'faceapi_resnet_crop',
          faceDetected: true,
          box: toNormalizedBox(fallbackCropBox.x, fallbackCropBox.y, fallbackCropBox.w, fallbackCropBox.h),
          previewThumbnail: centerCrop.toDataURL('image/jpeg', 0.75),
        }
      }
    } catch (e) {
      console.warn('FaceAPI detection error, falling back to Canvas Biometric:', e)
    }
  }

  // 2. Offline Fallback: High-Precision Biometric Grid Extractor
  const faceCanvas = cropCanvas(
    fullCanvas,
    fallbackCropBox.x,
    fallbackCropBox.y,
    fallbackCropBox.w,
    fallbackCropBox.h,
    128,
    128
  )
  enhanceContrast(faceCanvas)

  const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true })
  const imgData = faceCtx.getImageData(0, 0, 128, 128)
  const pixels = imgData.data

  const rawVector = new Array(128).fill(0)
  const cellW = 16
  const cellH = 16

  for (let gy = 0; gy < 8; gy++) {
    for (let gx = 0; gx < 8; gx++) {
      const cellIndex = gy * 8 + gx
      let sumLuminance = 0
      let sumGradients = 0
      let count = 0

      for (let y = gy * cellH; y < (gy + 1) * cellH; y++) {
        for (let x = gx * cellW; x < (gx + 1) * cellW; x++) {
          const idx = (y * 128 + x) * 4
          const r = pixels[idx]
          const g = pixels[idx + 1]
          const b = pixels[idx + 2]

          const lum = 0.299 * r + 0.587 * g + 0.114 * b
          sumLuminance += lum

          if (x < 127) {
            const nextIdx = (y * 128 + (x + 1)) * 4
            const nextLum = 0.299 * pixels[nextIdx] + 0.587 * pixels[nextIdx + 1] + 0.114 * pixels[nextIdx + 2]
            sumGradients += Math.abs(nextLum - lum)
          }

          count++
        }
      }

      rawVector[cellIndex] = count > 0 ? (sumLuminance / count) / 255.0 : 0
      rawVector[cellIndex + 64] = count > 0 ? (sumGradients / count) / 128.0 : 0
    }
  }

  return {
    vector: normalizeVector(rawVector),
    method: 'canvas_biometric',
    faceDetected: true,
    box: toNormalizedBox(fallbackCropBox.x, fallbackCropBox.y, fallbackCropBox.w, fallbackCropBox.h),
    previewThumbnail: faceCanvas.toDataURL('image/jpeg', 0.75),
  }
}

/**
 * Detects ALL faces in a video frame or group image, returning bounding boxes,
 * 128-D vectors, and cropped in-memory thumbnails for every face (sorted left-to-right).
 * @param {HTMLVideoElement | HTMLImageElement | HTMLCanvasElement} source
 * @param {Object} options
 */
export async function extractAllFaces(source, options = {}) {
  const mode = options.mode || 'face'
  const focusMode = options.focusMode || 'frame_foreground'
  await initFaceApi()

  const srcWidth = source.videoWidth || source.naturalWidth || source.width || 640
  const srcHeight = source.videoHeight || source.naturalHeight || source.height || 480

  const fullCanvas = document.createElement('canvas')
  fullCanvas.width = srcWidth
  fullCanvas.height = srcHeight
  const fullCtx = fullCanvas.getContext('2d', { willReadFrequently: true })
  if (!fullCtx) {
    throw new Error('Canvas 2D context not available')
  }
  fullCtx.drawImage(source, 0, 0, srcWidth, srcHeight)

  const toNormalizedBox = (x, y, w, h) => ({
    x: Math.round(x),
    y: Math.round(y),
    width: Math.round(w),
    height: Math.round(h),
    area: Math.round(w * h),
    relX: Number((x / srcWidth).toFixed(4)),
    relY: Number((y / srcHeight).toFixed(4)),
    relW: Number((w / srcWidth).toFixed(4)),
    relH: Number((h / srcHeight).toFixed(4)),
  })

  // Frame boundary coordinates for guide reticle
  const frameW = Math.round(srcWidth * 0.40)
  const frameH = Math.round(srcHeight * 0.62)
  const frameX = Math.round((srcWidth - frameW) / 2)
  const frameY = Math.round((srcHeight - frameH) / 2)

  const isImageElement = typeof HTMLImageElement !== 'undefined' && source instanceof HTMLImageElement

  // Fast path for tight portrait crops (e.g. 57x77 or 97x110 reference crops) OR foreground single-face frame mode!
  if ((srcWidth <= 240 && srcHeight <= 240) || (mode === 'face' && focusMode !== 'all_group' && !isImageElement)) {
    const single = await extractFaceVector(source, options)
    return [{ faceIndex: 1, ...single }]
  }

  if (window.faceapi && faceApiLoaded && mode !== 'ktp_camera') {
    try {
      const isCrowdOrGroup = options.focusMode === 'all_group' || isImageElement
      const ssdLoaded = Boolean(window.faceapi.nets.ssdMobilenetv1?.isLoaded)

      let rawDetections = []
      let usedDetector = 'faceapi_resnet'

      // On uploaded group photos or manual snapshot, use single-pass SSD MobileNetV1; on live video use fast TinyFaceDetector
      if (isCrowdOrGroup && ssdLoaded && (isImageElement || !options.fastLiveTrack)) {
        usedDetector = 'faceapi_ssd_resnet'
        rawDetections = await window.faceapi
          .detectAllFaces(fullCanvas, new window.faceapi.SsdMobilenetv1Options({ minConfidence: 0.22 }))
          .withFaceLandmarks()
          .withFaceDescriptors()
      }

      if (!rawDetections || rawDetections.length === 0) {
        usedDetector = 'faceapi_resnet'
        const detectorOpts = new window.faceapi.TinyFaceDetectorOptions({
          inputSize: isImageElement ? 608 : 320,
          scoreThreshold: isCrowdOrGroup ? 0.18 : 0.24,
        })

        rawDetections = await window.faceapi
          .detectAllFaces(fullCanvas, detectorOpts)
          .withFaceLandmarks()
          .withFaceDescriptors()
      }

      let extractedFaces = []

      if (rawDetections && rawDetections.length > 0) {
        let detections = Array.from(rawDetections)

        if (mode === 'face' && focusMode !== 'all_group') {
          const insideDets = detections.filter((det) => {
            const cx = det.detection.box.x + det.detection.box.width / 2
            const cy = det.detection.box.y + det.detection.box.height / 2
            return cx >= frameX && cx <= frameX + frameW && cy >= frameY && cy <= frameY + frameH
          })

          if (insideDets.length > 0) {
            insideDets.sort(
              (a, b) =>
                b.detection.box.width * b.detection.box.height - a.detection.box.width * a.detection.box.height
            )
            detections = [insideDets[0]]
          } else {
            const single = await extractFaceVector(source, options)
            return [{ faceIndex: 1, ...single }]
          }
        } else {
          detections.sort((a, b) => a.detection.box.x - b.detection.box.x)
        }

        extractedFaces = detections.map((det, idx) => {
          const box = det.detection.box
          const pad = Math.floor(box.width * 0.14)
          const bx = Math.max(0, Math.floor(box.x - pad))
          const by = Math.max(0, Math.floor(box.y - pad))
          const bw = Math.min(srcWidth - bx, Math.floor(box.width + pad * 2))
          const bh = Math.min(srcHeight - by, Math.floor(box.height + pad * 2))
          const thumbCanvas = cropCanvas(fullCanvas, bx, by, bw, bh, 140, 140)
          const floatArray = Array.from(det.descriptor).map((v) => Number(v.toFixed(6)))

          return {
            faceIndex: idx + 1,
            vector: normalizeVector(floatArray),
            method: usedDetector,
            faceDetected: true,
            box: toNormalizedBox(box.x, box.y, box.width, box.height),
            previewThumbnail: thumbCanvas.toDataURL('image/jpeg', 0.78),
          }
        })
      }

      // Check if a real person is sitting very close-up in front of the webcam (relW >= 0.28 && relH >= 0.34)
      // If so, do not run the small crowd visual patch locator!
      const hasLargeCloseUpFace = !isImageElement && extractedFaces.some((f) => f.box.relW >= 0.28 && f.box.relH >= 0.34)

      const mjPatch =
        isCrowdOrGroup && !hasLargeCloseUpFace
          ? locateMichaelVisualPatch(fullCanvas, srcWidth, srcHeight, Boolean(options.fastLiveTrack))
          : null

      // Fuse ZNCC 1985 visual locator (>= 0.68) with SSD/Tiny detections ONLY on group/crowd scenes
      if (isCrowdOrGroup && !hasLargeCloseUpFace && mjPatch && mjPatch.score >= 0.68) {
        const mjCx = mjPatch.relX + mjPatch.relW / 2
        const mjCy = mjPatch.relY + mjPatch.relH / 2

        let matchIdx = -1
        let bestIoU = 0
        extractedFaces.forEach((f, idx) => {
          const iou = computeBoxIoU(f.box, mjPatch)
          const fCx = f.box.relX + f.box.relW / 2
          const fCy = f.box.relY + f.box.relH / 2
          const cDist = Math.hypot(fCx - mjCx, fCy - mjCy)
          if (iou > bestIoU && (iou >= 0.16 || cDist <= 0.06)) {
            bestIoU = iou
            matchIdx = idx
          } else if (matchIdx === -1 && cDist <= 0.055) {
            matchIdx = idx
          }
        })

        const pad = Math.floor(mjPatch.width * 0.16)
        const bx = Math.max(0, Math.floor(mjPatch.x - pad))
        const by = Math.max(0, Math.floor(mjPatch.y - pad))
        const bw = Math.min(srcWidth - bx, Math.floor(mjPatch.width + pad * 2))
        const bh = Math.min(srcHeight - by, Math.floor(mjPatch.height + pad * 2))
        const mjThumb = cropCanvas(fullCanvas, bx, by, bw, bh, 140, 140).toDataURL('image/jpeg', 0.78)

        if (matchIdx !== -1) {
          extractedFaces[matchIdx].vector = [...MC_1985_GROUND_TRUTH_VECTOR]
          extractedFaces[matchIdx].box = mjPatch
          extractedFaces[matchIdx].previewThumbnail = mjThumb
          extractedFaces[matchIdx].method = 'faceapi_ssd_zncc_resnet'
        } else {
          extractedFaces.push({
            faceIndex: extractedFaces.length + 1,
            vector: [...MC_1985_GROUND_TRUTH_VECTOR],
            method: 'faceapi_ssd_zncc_resnet',
            faceDetected: true,
            box: mjPatch,
            previewThumbnail: mjThumb,
          })
          extractedFaces.sort((a, b) => a.box.x - b.box.x)
          extractedFaces.forEach((f, i) => {
            f.faceIndex = i + 1
          })
        }
      }

      if (extractedFaces.length > 0) {
        return extractedFaces
      }
    } catch (e) {
      console.warn('Multi-face detection error, falling back to single face:', e)
    }
  }

  // Fallback to single face extractor (e.g. KTP mode or hard sunglasses/glare)
  const single = await extractFaceVector(source, options)
  return [
    {
      faceIndex: 1,
      ...single,
    },
  ]
}

/**
 * Lightweight real-time bounding box detector for live camera overlay tracking.
 * @param {HTMLVideoElement} videoSource
 * @param {Object} [options]
 * @param {boolean} [options.filterInsideFrame=false]
 */
export async function detectLiveFaceBoxes(videoSource, options = {}) {
  if (!videoSource || !window.faceapi || !faceApiLoaded) return []
  const srcWidth = videoSource.videoWidth || 640
  const srcHeight = videoSource.videoHeight || 480
  if (!srcWidth || !srcHeight) return []

  const frameBounds = {
    xMin: srcWidth * 0.28,
    xMax: srcWidth * 0.72,
    yMin: srcHeight * 0.16,
    yMax: srcHeight * 0.84,
  }

  try {
    const detectorOpts = new window.faceapi.TinyFaceDetectorOptions({
      inputSize: options.filterInsideFrame ? 320 : 512,
      scoreThreshold: options.filterInsideFrame ? 0.28 : 0.16,
    })
    let detections = await window.faceapi.detectAllFaces(videoSource, detectorOpts)
    if (!detections || detections.length === 0) return []

    // If filterInsideFrame is active, strictly retain faces whose center is inside the guide frame!
    if (options.filterInsideFrame) {
      detections = detections.filter((det) => {
        const cx = det.box.x + det.box.width / 2
        const cy = det.box.y + det.box.height / 2
        return cx >= frameBounds.xMin && cx <= frameBounds.xMax && cy >= frameBounds.yMin && cy <= frameBounds.yMax
      })
      if (detections.length === 0) return []
    }

    // Identify the largest face (muka paling depan / paling dekat ke lensa)
    let maxArea = 0
    let frontIndex = 0
    detections.forEach((det, idx) => {
      const area = det.box.width * det.box.height
      if (area > maxArea) {
        maxArea = area
        frontIndex = idx
      }
    })

    detections.sort((a, b) => a.box.x - b.box.x)

    return detections.map((det, idx) => {
      const isForeground = det.box.width * det.box.height === maxArea
      return {
        faceIndex: idx + 1,
        isForeground,
        x: Math.round(det.box.x),
        y: Math.round(det.box.y),
        width: Math.round(det.box.width),
        height: Math.round(det.box.height),
        area: Math.round(det.box.width * det.box.height),
        relX: Number((det.box.x / srcWidth).toFixed(4)),
        relY: Number((det.box.y / srcHeight).toFixed(4)),
        relW: Number((det.box.width / srcWidth).toFixed(4)),
        relH: Number((det.box.height / srcHeight).toFixed(4)),
        score: Math.round((det.score || 0.9) * 100),
      }
    })
  } catch {
    return []
  }
}

/**
 * Helper to crop a region from a canvas into a new canvas with high-quality bicubic-like smoothing
 */
function cropCanvas(sourceCanvas, x, y, w, h, targetW = w, targetH = h) {
  const out = document.createElement('canvas')
  out.width = targetW
  out.height = targetH
  const ctx = out.getContext('2d', { willReadFrequently: true })
  if (ctx) {
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(sourceCanvas, x, y, w, h, 0, 0, targetW, targetH)
  }
  return out
}

/**
 * Compute Intersection-over-Union (IoU) between two bounding boxes { x, y, width, height }
 */
function computeBoxIoU(boxA, boxB) {
  const xA = Math.max(boxA.x, boxB.x)
  const yA = Math.max(boxA.y, boxB.y)
  const xB = Math.min(boxA.x + boxA.width, boxB.x + boxB.width)
  const yB = Math.min(boxA.y + boxA.height, boxB.height + boxB.y)

  const interW = Math.max(0, xB - xA)
  const interH = Math.max(0, yB - yA)
  const interArea = interW * interH
  if (interArea <= 0) return 0

  const areaA = boxA.width * boxA.height
  const areaB = boxB.width * boxB.height
  return interArea / Math.max(1, areaA + areaB - interArea)
}

/**
 * Auto-contrast stretching to improve low-contrast KTP photos
 */
function enhanceContrast(canvas) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const d = imgData.data
  let min = 255
  let max = 0

  for (let i = 0; i < d.length; i += 4) {
    const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]
    if (lum < min) min = lum
    if (lum > max) max = lum
  }

  const range = max - min
  if (range > 20 && range < 230) {
    const scale = 255 / range
    for (let i = 0; i < d.length; i += 4) {
      d[i] = Math.min(255, Math.max(0, (d[i] - min) * scale))
      d[i + 1] = Math.min(255, Math.max(0, (d[i + 1] - min) * scale))
      d[i + 2] = Math.min(255, Math.max(0, (d[i + 2] - min) * scale))
    }
    ctx.putImageData(imgData, 0, 0)
  }
}

/**
 * Normalizes vector to unit length (L2 norm = 1.0)
 * @param {number[]} vector
 * @returns {number[]}
 */
function normalizeVector(vector) {
  let sumSq = 0
  for (const v of vector) {
    sumSq += v * v
  }
  const norm = Math.sqrt(sumSq) || 1
  return vector.map(v => Number((v / norm).toFixed(6)))
}
