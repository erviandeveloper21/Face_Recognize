<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { extractFaceVector, extractAllFaces, detectLiveFaceBoxes, initFaceApi, matchFacesLocally } from './utils/faceExtractor'

// Navigation & State
const activeTab = ref('camera')
const profiles = ref([])
const logs = ref([])
const storageMetrics = ref(null)
const isLoading = ref(false)
const apiError = ref(null)
const apiSuccess = ref(null)

// Camera State
const videoRef = ref(null)
const overlayCanvasRef = ref(null)
const isCameraActive = ref(false)
const cameraLoading = ref(false)
const cameraStream = ref(null)
const isAutoScan = ref(false)
let autoScanInterval = null
let liveBoxInterval = null

// Focus & Tracking Mode: 'frame_foreground' (Hanya di Frame & Muka Paling Depan) | 'all_group' (Deteksi Semua Orang di Group)
const focusMode = ref('frame_foreground')
const groupBoxDisplayMode = ref('all') // 'all' (Tampilkan Semua: Kotak Hijau Cocok & Kotak Merah Tidak Cocok) | 'highest_only' (1 Kotak Tertinggi Saja)
const isPeakLockEnabled = ref(true) // Otomatis kunci (lock) wajah yang COCOK (>=70%) pada kotak hijau & TIDAK COCOK (<70%) pada kotak merah
const lockedPeakFace = ref(null)
const lockedUnmatchedFace = ref(null)
let lockedPeakTimestamp = 0
let lastAnnotatedImg = null

const resetPeakLock = () => {
  lockedPeakFace.value = null
  lockedUnmatchedFace.value = null
  lockedPeakTimestamp = 0
  prevWinnerBox = null
}

const isFacePassed = (faceItem) => {
  const pct = Number(faceItem?.top_candidate?.similarity_percentage ?? 0)
  return Boolean(faceItem?.is_match || pct >= 70)
}

const getWinningFace = (facesList) => {
  if (!facesList || facesList.length === 0) return null

  // 1. Prioritaskan wajah yang LULUS / COCOK (is_match && >= 70%) dan urutkan dari PERSENTASE PALING BESAR!
  const matchedFaces = facesList.filter(
    (o) => Boolean(o.is_match) && Number(o.top_candidate?.similarity_percentage ?? 0) >= 70
  )
  if (matchedFaces.length > 0) {
    matchedFaces.sort((a, b) => {
      const pA = Number(a.top_candidate?.similarity_percentage ?? 0)
      const pB = Number(b.top_candidate?.similarity_percentage ?? 0)
      if (pB !== pA) return pB - pA
      const dA = Number(a.top_candidate?.distance ?? 999)
      const dB = Number(b.top_candidate?.distance ?? 999)
      return dA - dB
    })
    return matchedFaces[0]
  }

  // 2. Jika belum ada yang lulus >= 70%, ambil yang persentasenya tertinggi (akan ditampilkan kuning/oranye, BUKAN kotak hijau)
  let maxPct = -1
  let minDist = 999
  let highestFace = null
  facesList.forEach((o) => {
    const p = Number(o.top_candidate?.similarity_percentage ?? 0)
    const d = Number(o.top_candidate?.distance ?? 999)
    if (p > maxPct || (p === maxPct && d < minDist)) {
      maxPct = p
      minDist = d
      highestFace = o
    }
  })
  return highestFace || facesList[0]
}

const isMichaelTarget = (faceItem) => {
  const candName = (faceItem?.top_candidate?.name || '').toLowerCase()
  const pct = Number(faceItem?.top_candidate?.similarity_percentage ?? 0)
  return Boolean(faceItem?.is_michael || (candName.includes('michael') && faceItem?.is_match && pct >= 70))
}

// Bounding Box & Group Comparison State
const liveBoxes = ref([])
const recognizedFacesList = ref([])
const annotatedPhotoDataUrl = ref(null)
const recognizePhotoFileName = ref('')

const isWinningTargetFace = (faceItem) => {
  const winner = getWinningFace(recognizedFacesList.value)
  if (!winner) return false
  return faceItem.face_index === winner.face_index
}

const sortedRecognizedFaces = computed(() => {
  if (!recognizedFacesList.value || recognizedFacesList.value.length === 0) return []
  return [...recognizedFacesList.value].sort((a, b) => {
    const aWin = isWinningTargetFace(a) ? 1 : 0
    const bWin = isWinningTargetFace(b) ? 1 : 0
    if (aWin !== bWin) return bWin - aWin
    const aPass = isFacePassed(a) ? 1 : 0
    const bPass = isFacePassed(b) ? 1 : 0
    if (aPass !== bPass) return bPass - aPass
    const aPct = a.top_candidate?.similarity_percentage ?? 0
    const bPct = b.top_candidate?.similarity_percentage ?? 0
    return bPct - aPct
  })
})

const matchedFacesCount = computed(() =>
  recognizedFacesList.value.filter((f) => isFacePassed(f)).length
)

const unmatchedFacesCount = computed(() =>
  recognizedFacesList.value.filter((f) => !isFacePassed(f)).length
)

// Live Camera Recognition State
const cameraRegisterName = ref('')
const cameraRegisterRole = ref('Karyawan')
const registerSource = ref('ktp') // 'camera' | 'ktp'
const cameraFrameMode = ref('face') // 'face' | 'ktp'
const ktpFileName = ref('')
const ktpVector = ref(null)
const ktpPreviewThumbnail = ref(null)
const ktpProcessing = ref(false)
const liveRecognitionResult = ref(null)
const lastExtractedSample = ref([])
const extractionMethod = ref('')
const cameraCaptureThumbnail = ref(null)

const applyRecognitionFaces = (json, extractedFaces) => {
  const firstFace = extractedFaces[0]
  if (json.faces && Array.isArray(json.faces)) {
    recognizedFacesList.value = json.faces.map((f, idx) => ({
      ...f,
      previewThumbnail: extractedFaces[(f.face_index || idx + 1) - 1]?.previewThumbnail || firstFace.previewThumbnail,
    }))
  } else {
    recognizedFacesList.value = [
      {
        ...json,
        face_index: 1,
        box: firstFace.box,
        previewThumbnail: firstFace.previewThumbnail,
      },
    ]
  }

  const winningFace = getWinningFace(recognizedFacesList.value)
  if (winningFace) {
    if (isFacePassed(winningFace)) {
      const curPct = Number(winningFace.top_candidate?.similarity_percentage ?? 0)
      const prevPct = Number(lockedPeakFace.value?.top_candidate?.similarity_percentage ?? -1)
      if (!lockedPeakFace.value || curPct >= prevPct) {
        lockedPeakFace.value = {
          ...winningFace,
          top_candidate: { ...winningFace.top_candidate },
        }
      }
      lockedUnmatchedFace.value = null
    } else if (!lockedPeakFace.value) {
      lockedUnmatchedFace.value = {
        ...winningFace,
        top_candidate: { ...winningFace.top_candidate },
      }
    }
  }
  if (winningFace?.previewThumbnail) {
    cameraCaptureThumbnail.value = winningFace.previewThumbnail
    const winExtract = extractedFaces[(winningFace.face_index || 1) - 1]
    if (winExtract?.vector) {
      lastExtractedSample.value = winExtract.vector.slice(0, 8)
    }
  }
}

// Simulation controls
const selectedProfileId = ref('')
const selectedCondition = ref('normal')
const thresholdValue = ref(0.68)
const lastVisitResult = ref(null)

// New profile registration form (manual)
const newName = ref('')
const newIdentifier = ref('')
const newRole = ref('Staff')
const newGender = ref('Male')

const conditions = [
  { key: 'normal', label: '☀️ Kedatangan Normal (Pencahayaan & Posisi Standar)' },
  { key: 'slight_angle', label: '🔄 Wajah Menoleh / Tilted (+/- 15°)' },
  { key: 'lighting_variation', label: '💡 Pencahayaan Berbeda (Redup / Backlight)' },
  { key: 'wearing_glasses', label: '👓 Menggunakan Aksesoris (Kacamata / Topi)' },
  { key: 'fatigue_expression', label: '😐 Perubahan Ekspresi (Tersenyum / Serius)' },
  { key: 'unknown_stranger', label: '👤 Orang Asing (Belum Terdaftar / Penyusup)' },
]

const roles = ['Karyawan', 'Staff', 'Engineer', 'Manager', 'Director', 'Tamu', 'Security']

// Helper for percentage coloring and status text
const percentageColor = (val) => {
  const n = Number(val) || 0
  if (n >= 70) return '#10b981'
  if (n >= 50) return '#f59e0b'
  return '#ef4444'
}

const similarityStatusText = (val, isMatch) => {
  const n = Number(val) || 0
  if (isMatch || n >= 70) {
    if (n >= 90) return '🌟 Sangat Identik (Kemiripan Tinggi)'
    return '✅ Wajah Cocok (Dikenali)'
  }
  if (n >= 50) return '⚠️ Kemiripan Sedang (Kurang Mirip)'
  return '🚫 Tidak Cocok (Beda Orang)'
}

// Draw bounding boxes (Pengkotakan Wajah) on the camera overlay canvas
const drawOverlayBoxes = () => {
  const canvas = overlayCanvasRef.value
  const video = videoRef.value
  if (!canvas || !video || !isCameraActive.value) return

  const cw = video.clientWidth || 640
  const ch = video.clientHeight || 480
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw
    canvas.height = ch
  }

  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, cw, ch)

  // Compute exact rendered video rectangle inside object-fit: cover container so boxes never stretch or drift
  const vw = video.videoWidth || cw
  const vh = video.videoHeight || ch
  const coverScale = Math.max(cw / vw, ch / vh)
  const renderedW = vw * coverScale
  const renderedH = vh * coverScale
  const offsetX = (cw - renderedW) / 2
  const offsetY = (ch - renderedH) / 2

  const isMirrored = cameraFrameMode.value !== 'ktp' && focusMode.value !== 'all_group'

  // Determine boxes to draw based on focusMode ('frame_foreground' vs 'all_group')
  let boxesToDraw = []

  if (recognizedFacesList.value.length > 0 && !annotatedPhotoDataUrl.value) {
    const winningFace = getWinningFace(recognizedFacesList.value)
    const winnerIndex = winningFace?.face_index
    const hasAnyMichael = Boolean(winningFace && isMichaelTarget(winningFace))

    boxesToDraw = recognizedFacesList.value.map((f) => {
      const pct = f.top_candidate?.similarity_percentage ?? 0
      const candName = f.top_candidate?.name || 'Unknown'
      const isWinner = winnerIndex != null && f.face_index === winnerIndex
      const isPassed = Boolean(f.is_match || Number(pct) >= 70)
      const isThisMichael = isWinner && isPassed && isMichaelTarget(f)

      let label = ''
      let color = '#64748b' // Default selain yang tertinggi: abu-abu netral

      if (focusMode.value === 'all_group') {
        if (isWinner && isPassed) {
          // Kotak HIJAU HANYA untuk 1 wajah COCOK dengan persentase PALING BESAR!
          color = '#10b981'
          label = isThisMichael
            ? `🔒 INI MICHAEL (#${f.face_index}: ${pct}%)`
            : `🔒 COCOK TERTINGGI: ${candName} (#${f.face_index}: ${pct}%)`
        } else if (!isPassed) {
          // Kotak MERAH untuk semua wajah yang TIDAK COCOK (< 70%)
          color = '#ef4444'
          label = isWinner
            ? `🔒 ❌ TIDAK COCOK (#${f.face_index}: ${pct}%)`
            : `❌ TIDAK COCOK (#${f.face_index}: ${pct}%)`
        } else {
          // Wajah lain yang >= 70% namun bukan persentase tertinggi
          color = '#f59e0b'
          label = `#${f.face_index}: ${candName} (${pct}%)`
        }
      } else {
        if (isWinner && isPassed) {
          color = '#10b981'
          label = isThisMichael ? `🔒 INI MICHAEL (${pct}%)` : `🔒 COCOK: ${candName} (${pct}%)`
        } else if (!isPassed) {
          color = '#ef4444'
          label = `🔒 ❌ TIDAK COCOK: Bukan ${candName} (${pct}%)`
        } else {
          color = '#f59e0b'
          label = `Wajah #${f.face_index} (${pct}%)`
        }
      }

      return {
        faceIndex: f.face_index,
        isForeground: f.box?.isForeground ?? true,
        relX: f.box?.relX ?? 0.25,
        relY: f.box?.relY ?? 0.15,
        relW: f.box?.relW ?? 0.5,
        relH: f.box?.relH ?? 0.7,
        label,
        color,
        isMatch: isWinner && isPassed,
      }
    })
  } else {
    // Live tracking before recognition completes (cyan, bukan hijau sampai terbukti cocok)
    boxesToDraw = liveBoxes.value.map((b) => ({
      faceIndex: b.faceIndex,
      isForeground: b.isForeground,
      relX: b.relX,
      relY: b.relY,
      relW: b.relW,
      relH: b.relH,
      label: b.isForeground ? `🔍 Memindai Wajah...` : `Wajah #${b.faceIndex}`,
      color: '#38bdf8',
      isMatch: false,
    }))
  }

  // In frame_foreground or highest_only mode, strictly draw ONLY 1 box (the locked highest matching face)
  if (boxesToDraw.length > 1 && (focusMode.value === 'frame_foreground' || groupBoxDisplayMode.value === 'highest_only')) {
    const highestBox = boxesToDraw.find((b) => b.isMatch) || boxesToDraw.find((b) => b.isForeground) || boxesToDraw[0]
    boxesToDraw = [highestBox]
  }

  boxesToDraw.forEach((b) => {
    const bw = b.relW * renderedW
    const bh = b.relH * renderedH
    const effRelX = isMirrored ? 1 - b.relX - b.relW : b.relX
    const bx = offsetX + effRelX * renderedW
    const by = offsetY + b.relY * renderedH

    // Main bounding box rectangle with glow if matched target
    ctx.strokeStyle = b.color
    ctx.lineWidth = b.isMatch ? 3.5 : 2.5
    ctx.strokeRect(bx, by, bw, bh)

    // Corner brackets
    const cornerLen = Math.min(20, bw * 0.22, bh * 0.22)
    ctx.lineWidth = 4.5
    ctx.beginPath()
    // Top-left
    ctx.moveTo(bx, by + cornerLen)
    ctx.lineTo(bx, by)
    ctx.lineTo(bx + cornerLen, by)
    // Top-right
    ctx.moveTo(bx + bw - cornerLen, by)
    ctx.lineTo(bx + bw, by)
    ctx.lineTo(bx + bw, by + cornerLen)
    // Bottom-left
    ctx.moveTo(bx, by + bh - cornerLen)
    ctx.lineTo(bx, by + bh)
    ctx.lineTo(bx + cornerLen, by + bh)
    // Bottom-right
    ctx.moveTo(bx + bw - cornerLen, by + bh)
    ctx.lineTo(bx + bw, by + bh)
    ctx.lineTo(bx + bw, by + bh - cornerLen)
    ctx.stroke()

    // Top label badge
    ctx.font = 'bold 12px system-ui, -apple-system, sans-serif'
    const textWidth = ctx.measureText(b.label).width
    const padX = 8
    const badgeH = 22
    const badgeW = textWidth + padX * 2
    const badgeX = Math.max(2, Math.min(cw - badgeW - 2, bx))
    const badgeY = by > badgeH + 4 ? by - badgeH - 2 : by + 2

    ctx.fillStyle = b.color
    ctx.beginPath()
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 5)
    ctx.fill()

    ctx.fillStyle = '#ffffff'
    ctx.fillText(b.label, badgeX + padX, badgeY + 15)
  })
}

const isAutoRecognizing = ref(false)
let liveAutoRecognizeInterval = null
let liveLoopActive = false
let lastBackendSyncTime = 0
let prevWinnerBox = null

const smoothBox = (newBox, oldBox) => {
  if (!newBox) return oldBox
  if (!oldBox) return newBox
  const dist = Math.hypot((newBox.relX || 0) - (oldBox.relX || 0), (newBox.relY || 0) - (oldBox.relY || 0))
  if (dist > 0.22) return newBox
  const alpha = 0.62
  return {
    ...newBox,
    relX: Number((alpha * newBox.relX + (1 - alpha) * oldBox.relX).toFixed(4)),
    relY: Number((alpha * newBox.relY + (1 - alpha) * oldBox.relY).toFixed(4)),
    relW: Number((alpha * newBox.relW + (1 - alpha) * oldBox.relW).toFixed(4)),
    relH: Number((alpha * newBox.relH + (1 - alpha) * oldBox.relH).toFixed(4)),
  }
}

const startLiveBoxTracking = () => {
  stopLiveBoxTracking()
  liveLoopActive = true

  const runResponsiveLiveLoop = async () => {
    while (liveLoopActive) {
      if (
        !isCameraActive.value ||
        !videoRef.value ||
        isLoading.value ||
        annotatedPhotoDataUrl.value
      ) {
        await new Promise((r) => setTimeout(r, 180))
        continue
      }

      try {
        isAutoRecognizing.value = true
        const activeExtractMode = cameraFrameMode.value === 'ktp' ? 'ktp_camera' : 'face'
        let extractedFaces = await extractAllFaces(videoRef.value, {
          mode: activeExtractMode,
          focusMode: focusMode.value,
          fastLiveTrack: true,
        })

        if (extractedFaces && extractedFaces.length > 0 && !annotatedPhotoDataUrl.value) {
          if (focusMode.value === 'frame_foreground' && extractedFaces.length > 1) {
            extractedFaces.sort((a, b) => (b.box?.area || 0) - (a.box?.area || 0))
            extractedFaces = [extractedFaces[0]]
          }

          // Instant 0ms Client-Side Biometric Matching for immediate responsive bounding box
          const localJson = matchFacesLocally(
            extractedFaces,
            profiles.value,
            thresholdValue.value,
            focusMode.value
          )

          // Smooth winning box coordinates & LOCK the highest matching percentage (>= 70%) on the Green Box!
          const winner = getWinningFace(localJson.faces || [])
          const nowMs = Date.now()

          if (winner && winner.box) {
            winner.box = smoothBox(winner.box, prevWinnerBox)
            prevWinnerBox = winner.box

            const curPct = Number(winner.top_candidate?.similarity_percentage ?? 0)
            const curPassed = Boolean(winner.is_match && curPct >= 70)

            if (isPeakLockEnabled.value) {
              if (curPassed) {
                const lockedPct = Number(lockedPeakFace.value?.top_candidate?.similarity_percentage ?? -1)
                const samePerson =
                  lockedPeakFace.value &&
                  lockedPeakFace.value.top_candidate?.name === winner.top_candidate?.name

                if (!lockedPeakFace.value || curPct >= lockedPct || !samePerson) {
                  // Lock new peak match!
                  lockedPeakFace.value = {
                    ...winner,
                    top_candidate: { ...winner.top_candidate },
                    metrics: winner.metrics ? { ...winner.metrics } : null,
                    is_locked_peak: true,
                  }
                } else {
                  // Keep locked peak percentage on the green box while tracking new box position!
                  winner.top_candidate = { ...lockedPeakFace.value.top_candidate }
                  winner.metrics = lockedPeakFace.value.metrics ? { ...lockedPeakFace.value.metrics } : winner.metrics
                  winner.message = lockedPeakFace.value.message
                  winner.is_match = true
                  lockedPeakFace.value.box = winner.box
                }
                lockedUnmatchedFace.value = null
                lockedPeakTimestamp = nowMs
              } else if (lockedPeakFace.value && nowMs - lockedPeakTimestamp < 2200) {
                // Hold the green box lock during brief camera shake if face is still near locked position
                const distToLock = Math.hypot(
                  (winner.box.relX || 0) - (lockedPeakFace.value.box?.relX || 0),
                  (winner.box.relY || 0) - (lockedPeakFace.value.box?.relY || 0)
                )
                if (distToLock < 0.16) {
                  winner.top_candidate = { ...lockedPeakFace.value.top_candidate }
                  winner.metrics = lockedPeakFace.value.metrics ? { ...lockedPeakFace.value.metrics } : winner.metrics
                  winner.message = lockedPeakFace.value.message
                  winner.is_match = true
                  winner.is_michael = lockedPeakFace.value.is_michael
                  lockedPeakFace.value.box = winner.box
                } else {
                  lockedPeakFace.value = null
                  lockedUnmatchedFace.value = {
                    ...winner,
                    top_candidate: { ...winner.top_candidate },
                  }
                }
              } else {
                lockedPeakFace.value = null
                lockedUnmatchedFace.value = {
                  ...winner,
                  top_candidate: { ...winner.top_candidate },
                }
              }
            }
          } else {
            prevWinnerBox = null
            if (nowMs - lockedPeakTimestamp >= 2200) {
              lockedPeakFace.value = null
              lockedUnmatchedFace.value = null
            }
          }

          liveRecognitionResult.value = localJson
          applyRecognitionFaces(localJson, extractedFaces)
          drawOverlayBoxes()

          // Non-blocking background sync to backend audit log (throttled every 2.2s)
          const now = Date.now()
          if (now - lastBackendSyncTime > 2200 && profiles.value.length > 0) {
            lastBackendSyncTime = now
            fetch('/api/face/recognize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                faces: extractedFaces.map((f) => ({
                  embedding: f.vector,
                  box: f.box,
                })),
                threshold: thresholdValue.value,
                condition_note: focusMode.value === 'all_group' ? 'Live Group Scan' : 'Live Camera Foreground Scan',
              }),
            }).catch(() => {})
          }
        }
      } catch {
        // Keep loop alive
      } finally {
        isAutoRecognizing.value = false
      }

      await new Promise((r) => setTimeout(r, 45))
    }
  }

  runResponsiveLiveLoop()
}

const stopLiveBoxTracking = () => {
  liveLoopActive = false
  prevWinnerBox = null
  if (liveBoxInterval) {
    clearInterval(liveBoxInterval)
    liveBoxInterval = null
  }
  if (liveAutoRecognizeInterval) {
    clearInterval(liveAutoRecognizeInterval)
    liveAutoRecognizeInterval = null
  }
}

// Camera controls
const startCamera = async () => {
  cameraLoading.value = true
  apiError.value = null
  annotatedPhotoDataUrl.value = null
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user',
      },
      audio: false,
    })

    cameraStream.value = stream
    await nextTick()
    if (videoRef.value) {
      videoRef.value.srcObject = stream
      await videoRef.value.play()
    }
    isCameraActive.value = true
    startLiveBoxTracking()
  } catch (err) {
    apiError.value = 'Tidak dapat mengakses kamera: ' + (err.message || err.name)
    isCameraActive.value = false
  } finally {
    cameraLoading.value = false
  }
}

const stopCamera = () => {
  stopLiveBoxTracking()
  if (cameraStream.value) {
    cameraStream.value.getTracks().forEach((track) => track.stop())
    cameraStream.value = null
  }
  if (videoRef.value) {
    videoRef.value.srcObject = null
  }
  isCameraActive.value = false
  stopAutoScan()
}

const setRegisterSource = (source) => {
  registerSource.value = source
  cameraFrameMode.value = source === 'ktp' ? 'ktp' : 'face'
}

// 1. Live Camera: Register Face (Ambil Wajah & Simpan Vektor)
const captureAndRegisterFromCamera = async () => {
  if (!videoRef.value || !isCameraActive.value) {
    apiError.value = 'Nyalakan kamera terlebih dahulu.'
    return
  }

  if (!cameraRegisterName.value.trim()) {
    apiError.value = 'Masukkan nama Anda terlebih dahulu sebelum mengambil wajah.'
    return
  }

  cameraFrameMode.value = 'face'
  isLoading.value = true
  apiError.value = null
  apiSuccess.value = null

  try {
    // Extract 128-D vector directly from video frame (Live Face mode)
    const { vector, method, previewThumbnail } = await extractFaceVector(videoRef.value, { mode: 'face' })
    lastExtractedSample.value = vector.slice(0, 8)
    extractionMethod.value = method + ' (Live Face)'
    cameraCaptureThumbnail.value = previewThumbnail

    // Send only 128 numbers + metadata to backend (NO photo is saved to DB/storage!)
    const res = await fetch('/api/face/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: cameraRegisterName.value.trim(),
        role: cameraRegisterRole.value,
        embedding: vector,
      }),
    })

    const json = await res.json()
    if (json.status === 'success') {
      apiSuccess.value = `Sukses! Wajah "${cameraRegisterName.value}" berhasil disimpan ke sistem dalam bentuk 128 float angka (512 bytes). Foto dibuang & tidak disimpan.`
      cameraRegisterName.value = ''
      await refreshAll()
    } else {
      apiError.value = json.message || 'Gagal menyimpan wajah.'
    }
  } catch (err) {
    apiError.value = 'Error saat mengekstrak/mendaftarkan wajah: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// 1.A Scan KTP Langsung dari Bingkai Kamera
const captureKtpFromCamera = async () => {
  if (!videoRef.value || !isCameraActive.value) {
    apiError.value = 'Nyalakan kamera terlebih dahulu untuk scan kartu KTP.'
    return
  }

  cameraFrameMode.value = 'ktp'
  ktpProcessing.value = true
  apiError.value = null
  apiSuccess.value = null

  try {
    const { vector, previewThumbnail, method } = await extractFaceVector(videoRef.value, { mode: 'ktp_camera' })
    ktpVector.value = vector
    ktpPreviewThumbnail.value = previewThumbnail
    cameraCaptureThumbnail.value = previewThumbnail
    ktpFileName.value = 'Scan Kamera (Bingkai KTP)'
    lastExtractedSample.value = vector.slice(0, 8)
    extractionMethod.value = method + ' (Bingkai KTP Kamera)'
    apiSuccess.value = 'Pas foto pada Kartu KTP berhasil di-crop dari bingkai kamera & diekstrak menjadi 128 angka float! Silakan periksa preview di bawah lalu klik Simpan.'
  } catch (err) {
    apiError.value = 'Gagal mengekstrak pas foto KTP dari kamera: ' + err.message
  } finally {
    ktpProcessing.value = false
  }
}

// 1.B KTP File Scanner: Extract Face from KTP Photo & Discard KTP
const onKtpFileSelected = (event) => {
  const file = event.target.files?.[0]
  if (!file) return
  ktpFileName.value = file.name
  ktpProcessing.value = true
  apiError.value = null
  apiSuccess.value = null

  const reader = new FileReader()
  reader.onload = (e) => {
    const img = new Image()
    img.onload = async () => {
      try {
        const { vector, previewThumbnail, method } = await extractFaceVector(img, { mode: 'ktp_image' })
        ktpVector.value = vector
        ktpPreviewThumbnail.value = previewThumbnail
        cameraCaptureThumbnail.value = previewThumbnail
        lastExtractedSample.value = vector.slice(0, 8)
        extractionMethod.value = method + ' (Upload File KTP)'
        apiSuccess.value = `Wajah pada file KTP "${file.name}" berhasil dideteksi & diekstrak jadi 128 float angka! Foto KTP mentah langsung dibuang.`
      } catch (err) {
        apiError.value = 'Gagal mendeteksi wajah pada foto KTP: ' + err.message
      } finally {
        ktpProcessing.value = false
      }
    }
    img.src = e.target.result
  }
  reader.readAsDataURL(file)
}

const registerFromKtp = async () => {
  if (!ktpVector.value) {
    apiError.value = 'Silakan scan KTP lewat kamera atau pilih file foto KTP terlebih dahulu.'
    return
  }
  if (!cameraRegisterName.value.trim()) {
    apiError.value = 'Masukkan nama pemilik KTP terlebih dahulu.'
    return
  }

  isLoading.value = true
  apiError.value = null
  try {
    const res = await fetch('/api/face/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: cameraRegisterName.value.trim(),
        role: cameraRegisterRole.value,
        embedding: ktpVector.value,
      }),
    })
    const json = await res.json()
    if (json.status === 'success') {
      apiSuccess.value = `Sukses! Vektor wajah dari KTP "${cameraRegisterName.value}" berhasil didaftarkan ke MySQL (512 bytes). Bingkai kamera otomatis dialihkan ke Bingkai Wajah agar Anda bisa langsung tes verifikasi wajah asli!`
      cameraRegisterName.value = ''
      ktpVector.value = null
      ktpFileName.value = ''
      cameraFrameMode.value = 'face'
      await refreshAll()
    } else {
      apiError.value = json.message || 'Gagal menyimpan wajah KTP.'
    }
  } catch (err) {
    apiError.value = 'Error pendaftaran KTP: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// Helper: Draw bounding boxes & personil match labels onto an uploaded image canvas (in-memory only)
const buildAnnotatedImage = (img, facesResults) => {
  const maxW = 720
  const scale = img.naturalWidth > maxW ? maxW / img.naturalWidth : 1
  const cw = Math.round(img.naturalWidth * scale)
  const ch = Math.round(img.naturalHeight * scale)

  const canvas = document.createElement('canvas')
  canvas.width = cw
  canvas.height = ch
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.drawImage(img, 0, 0, cw, ch)

  // Cari SATU wajah pemenang (persentase kemiripan tertinggi di antara seluruh wajah)
  const highestFace = getWinningFace(facesResults)

  // JIKA MODE 'highest_only' (DEFAULT): HANYA TAMPILKAN SATU WAJAH SAJA YANG PALING TINGGI!
  const facesToRender = (groupBoxDisplayMode.value === 'highest_only' && highestFace)
    ? [highestFace]
    : facesResults

  facesToRender.forEach((f) => {
    const relX = f.box?.relX ?? 0.25
    const relY = f.box?.relY ?? 0.15
    const relW = f.box?.relW ?? 0.5
    const relH = f.box?.relH ?? 0.7

    const bx = relX * cw
    const by = relY * ch
    const bw = relW * cw
    const bh = relH * ch
    const pct = f.top_candidate?.similarity_percentage ?? 0
    const candName = f.top_candidate?.name || 'Unknown'
    const isWinner = Boolean(highestFace && f.face_index === highestFace.face_index)
    const isPassed = Boolean(f.is_match || Number(pct) >= 70)
    const isMichael = isWinner && isPassed && isMichaelTarget(f)

    let color = '#64748b' // default abu-abu netral untuk wajah lain
    let label = `#${f.face_index}: ${candName} (${pct}%)`

    if (isWinner && isPassed) {
      // HANYA SATU WAJAH TERTINGGI YANG LULUS THRESHOLD DIBERI KOTAK HIJAU TEBAL (LOCKED)!
      color = '#10b981'
      label = isMichael
        ? `🔒 INI MICHAEL (#${f.face_index}: ${pct}%)`
        : `🔒 COCOK TERTINGGI: ${candName} (#${f.face_index}: ${pct}%)`
    } else if (!isPassed) {
      // SEMUA WAJAH YANG TIDAK COCOK (< 70%) DIBERI KOTAK MERAH TEGAS!
      color = '#ef4444'
      label = isWinner
        ? `🔒 ❌ TIDAK COCOK (#${f.face_index}: ${pct}%)`
        : `❌ TIDAK COCOK (#${f.face_index}: ${pct}%)`
    } else {
      color = '#f59e0b'
      label = `#${f.face_index}: ${candName} (${pct}%)`
    }

    ctx.strokeStyle = color
    ctx.lineWidth = isWinner ? 4.5 : 2.4
    ctx.strokeRect(bx, by, bw, bh)

    // Corner brackets untuk semua kotak wajah (tebal 6px untuk pemenang, 3.5px untuk yang tidak cocok)
    const cornerLen = Math.min(20, bw * 0.25, bh * 0.25)
    ctx.lineWidth = isWinner ? 6 : 3.5
    ctx.beginPath()
    ctx.moveTo(bx, by + cornerLen)
    ctx.lineTo(bx, by)
    ctx.lineTo(bx + cornerLen, by)
    ctx.moveTo(bx + bw - cornerLen, by)
    ctx.lineTo(bx + bw, by)
    ctx.lineTo(bx + bw, by + cornerLen)
    ctx.moveTo(bx, by + bh - cornerLen)
    ctx.lineTo(bx, by + bh)
    ctx.lineTo(bx + cornerLen, by + bh)
    ctx.moveTo(bx + bw - cornerLen, by + bh)
    ctx.lineTo(bx + bw, by + bh)
    ctx.lineTo(bx + bw, by + bh - cornerLen)
    ctx.stroke()

    ctx.font = isWinner ? 'bold 13px system-ui, -apple-system, sans-serif' : '11px system-ui, -apple-system, sans-serif'
    const textW = ctx.measureText(label).width
    const padX = 8
    const badgeH = isWinner ? 24 : 20
    const badgeW = textW + padX * 2
    const badgeX = Math.max(2, Math.min(cw - badgeW - 2, bx))
    const badgeY = by > badgeH + 4 ? by - badgeH - 2 : by + 2

    ctx.fillStyle = color
    ctx.beginPath()
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 5)
    ctx.fill()

    ctx.fillStyle = '#ffffff'
    ctx.fillText(label, badgeX + padX, badgeY + (isWinner ? 16 : 14))
  })

  return canvas.toDataURL('image/jpeg', 0.85)
}

const setGroupBoxDisplayMode = (mode) => {
  groupBoxDisplayMode.value = mode
  if (lastAnnotatedImg && recognizedFacesList.value.length > 0) {
    annotatedPhotoDataUrl.value = buildAnnotatedImage(lastAnnotatedImg, recognizedFacesList.value)
  }
  drawOverlayBoxes()
}

// 2. Live Camera: Recognize Face / Group (Kenali & Pengkotakan Otomatis)
const captureAndRecognizeFromCamera = async (forcedMode = null) => {
  if (!videoRef.value || !isCameraActive.value) {
    apiError.value = 'Nyalakan kamera terlebih dahulu.'
    return
  }

  if (forcedMode === 'face') {
    cameraFrameMode.value = 'face'
  } else if (forcedMode === 'ktp_camera') {
    cameraFrameMode.value = 'ktp'
  }

  annotatedPhotoDataUrl.value = null
  const activeExtractMode = forcedMode || (cameraFrameMode.value === 'ktp' ? 'ktp_camera' : 'face')

  isLoading.value = true
  apiError.value = null

  try {
    // Extract faces in current video frame (strictly inside guide frame in foreground mode, or full scene in group mode)
    let extractedFaces = await extractAllFaces(videoRef.value, {
      mode: activeExtractMode,
      focusMode: focusMode.value,
    })

    // In frame_foreground mode, strictly take the largest face in front
    if (focusMode.value === 'frame_foreground' && extractedFaces.length > 1) {
      extractedFaces.sort((a, b) => (b.box?.area || 0) - (a.box?.area || 0))
      extractedFaces = [extractedFaces[0]]
    }

    const firstFace = extractedFaces[0]
    lastExtractedSample.value = firstFace.vector.slice(0, 8)
    extractionMethod.value =
      firstFace.method +
      (extractedFaces.length > 1
        ? ` (${extractedFaces.length} Wajah Terdeteksi)`
        : focusMode.value === 'frame_foreground'
          ? ' (Muka Paling Depan di Frame)'
          : activeExtractMode === 'ktp_camera'
            ? ' (Bingkai KTP)'
            : ' (Live Face)')
    cameraCaptureThumbnail.value = firstFace.previewThumbnail

    // Instant client-side biometric match
    const localJson = matchFacesLocally(extractedFaces, profiles.value, thresholdValue.value, focusMode.value)
    liveRecognitionResult.value = localJson
    applyRecognitionFaces(localJson, extractedFaces)
    drawOverlayBoxes()

    // Also sync with backend API if profiles exist
    if (profiles.value.length > 0) {
      try {
        const res = await fetch('/api/face/recognize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            faces: extractedFaces.map((f) => ({
              embedding: f.vector,
              box: f.box,
            })),
            threshold: thresholdValue.value,
            condition_note:
              activeExtractMode === 'ktp_camera'
                ? 'Scan Kartu KTP via Kamera'
                : extractedFaces.length > 1
                  ? `Live Group Scan (${extractedFaces.length} Wajah)`
                  : 'Live Webcam Face Scan',
          }),
        })
        const json = await res.json()
        if (json && json.faces && (!localJson.target_detected || json.target_face_index === localJson.target_face_index)) {
          liveRecognitionResult.value = json
          applyRecognitionFaces(json, extractedFaces)
          drawOverlayBoxes()
        }
        await Promise.all([fetchLogs(), fetchStorageMetrics()])
      } catch {
        // Keep local match result if backend is unreachable
      }
    }
  } catch (err) {
    apiError.value = 'Error saat mengenali wajah: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// 2.B Upload Foto / Foto Group untuk Tes Pengkotakan & Bandingkan Personil
const onRecognizePhotoSelected = (event) => {
  const file = event.target.files?.[0]
  if (!file) return

  recognizePhotoFileName.value = file.name
  isLoading.value = true
  apiError.value = null
  apiSuccess.value = null

  const reader = new FileReader()
  reader.onload = (e) => {
    const img = new Image()
    img.onload = async () => {
      try {
        const extractedFaces = await extractAllFaces(img, {
          mode: 'face',
          focusMode: 'all_group',
        })
        const firstFace = extractedFaces[0]
        lastExtractedSample.value = firstFace.vector.slice(0, 8)
        extractionMethod.value = `${firstFace.method} (${extractedFaces.length} Wajah di Foto)`
        cameraCaptureThumbnail.value = firstFace.previewThumbnail

        // Instant client-side multi-template match
        let finalJson = matchFacesLocally(extractedFaces, profiles.value, thresholdValue.value, 'all_group')

        if (profiles.value.length > 0) {
          try {
            const res = await fetch('/api/face/recognize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                faces: extractedFaces.map((f) => ({
                  embedding: f.vector,
                  box: f.box,
                })),
                threshold: thresholdValue.value,
                condition_note: `Upload Foto Test (${extractedFaces.length} Wajah)`,
              }),
            })
            const apiJson = await res.json()
            if (apiJson && apiJson.faces && (!finalJson.target_detected || apiJson.target_face_index === finalJson.target_face_index)) {
              finalJson = apiJson
            }
          } catch {
            // Keep instant local match if backend is offline
          }
        }

        liveRecognitionResult.value = finalJson
        applyRecognitionFaces(finalJson, extractedFaces)

        // Build annotated image with bounding boxes + matched personil name & %
        lastAnnotatedImg = img
        annotatedPhotoDataUrl.value = buildAnnotatedImage(img, recognizedFacesList.value)
        apiSuccess.value = `Berhasil mendeteksi & mengkotakkan ${recognizedFacesList.value.length} wajah pada foto "${file.name}"! Lihat hasil perbandingan dengan setiap personil di bawah.`
        await Promise.all([fetchLogs(), fetchStorageMetrics()])
      } catch (err) {
        apiError.value = 'Gagal memproses foto: ' + err.message
      } finally {
        isLoading.value = false
        event.target.value = ''
      }
    }
    img.src = e.target.result
  }
  reader.readAsDataURL(file)
}

// Auto scan toggle
const toggleAutoScan = () => {
  isAutoScan.value = !isAutoScan.value
  if (isAutoScan.value) {
    autoScanInterval = setInterval(() => {
      if (isCameraActive.value && !isLoading.value && profiles.value.length > 0) {
        captureAndRecognizeFromCamera()
      }
    }, 2500)
  } else {
    stopAutoScan()
  }
}

const stopAutoScan = () => {
  isAutoScan.value = false
  if (autoScanInterval) {
    clearInterval(autoScanInterval)
    autoScanInterval = null
  }
}

// Fetch helper functions
const fetchProfiles = async () => {
  try {
    const res = await fetch('/api/face/profiles')
    const json = await res.json()
    if (json.status === 'success') {
      profiles.value = json.data
      if (profiles.value.length > 0 && selectedProfileId.value === '' && profiles.value[0]) {
        selectedProfileId.value = profiles.value[0].id
      }
    }
  } catch (err) {
    apiError.value = 'Gagal memuat profil: ' + err.message
  }
}

const fetchLogs = async () => {
  try {
    const res = await fetch('/api/face/logs')
    const json = await res.json()
    if (json.status === 'success') {
      logs.value = json.data
    }
  } catch (err) {
    console.error(err)
  }
}

const fetchStorageMetrics = async () => {
  try {
    const res = await fetch('/api/face/storage-metrics')
    const json = await res.json()
    if (json.status === 'success') {
      storageMetrics.value = json.metrics
    }
  } catch (err) {
    console.error(err)
  }
}

const refreshAll = async () => {
  isLoading.value = true
  apiError.value = null
  try {
    await Promise.all([fetchProfiles(), fetchLogs(), fetchStorageMetrics()])
  } finally {
    isLoading.value = false
  }
}

// 1-Click Sync Multi-Era Michael Jackson Model (mc/model_michael_jackson.json)
const syncMcModel = async () => {
  isLoading.value = true
  apiError.value = null
  apiSuccess.value = null
  try {
    const res = await fetch('/api/face/import-mc-model', { method: 'POST' })
    const json = await res.json()
    if (json.status === 'success') {
      apiSuccess.value = json.message
      await refreshAll()
    } else {
      apiError.value = json.message || 'Gagal menyinkronkan model MC.'
    }
  } catch (err) {
    apiError.value = 'Koneksi error saat sinkronisasi model MC: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// 1-Click seed 5 faces (simulation)
const seedFiveFaces = async () => {
  isLoading.value = true
  apiError.value = null
  apiSuccess.value = null
  try {
    const res = await fetch('/api/face/simulate-seed', { method: 'POST' })
    const json = await res.json()
    if (json.status === 'success') {
      apiSuccess.value = json.message
      await refreshAll()
    } else {
      apiError.value = json.message || 'Gagal membuat simulasi muka.'
    }
  } catch (err) {
    apiError.value = 'Koneksi error: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// Simulate Next Visit ("Kedatangan Berikutnya")
const runVisitSimulation = async () => {
  isLoading.value = true
  apiError.value = null
  try {
    const payload = {
      condition: selectedCondition.value,
      threshold: thresholdValue.value,
    }
    if (selectedCondition.value !== 'unknown_stranger' && selectedProfileId.value) {
      payload.profile_id = selectedProfileId.value
    }

    const res = await fetch('/api/face/simulate-visit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const json = await res.json()
    if (json.recognition_result) {
      lastVisitResult.value = json
      await Promise.all([fetchProfiles(), fetchLogs(), fetchStorageMetrics()])
    } else {
      apiError.value = json.message || 'Gagal menjalankan simulasi kedatangan.'
    }
  } catch (err) {
    apiError.value = 'Error verifikasi wajah: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// Manual register profile
const registerProfile = async () => {
  if (!newName.value.trim()) {
    apiError.value = 'Nama wajib diisi.'
    return
  }

  isLoading.value = true
  apiError.value = null
  apiSuccess.value = null

  try {
    const res = await fetch('/api/face/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: newName.value.trim(),
        identifier: newIdentifier.value.trim() || undefined,
        role: newRole.value,
        gender: newGender.value,
        auto_generate_embedding: true,
      }),
    })

    const json = await res.json()
    if (json.status === 'success') {
      apiSuccess.value = json.message
      newName.value = ''
      newIdentifier.value = ''
      await refreshAll()
      activeTab.value = 'camera'
    } else {
      apiError.value = json.message || 'Gagal mendaftarkan profil.'
    }
  } catch (err) {
    apiError.value = 'Error pendaftaran: ' + err.message
  } finally {
    isLoading.value = false
  }
}

// Delete single profile
const deleteProfile = async (id) => {
  if (!confirm('Hapus profil ini dari database vektor?')) return
  try {
    await fetch(`/api/face/profiles/${id}`, { method: 'DELETE' })
    await refreshAll()
  } catch (err) {
    apiError.value = err.message
  }
}

// Reset/Clear all data
const clearAllData = async () => {
  if (!confirm('Hapus semua profil wajah dan riwayat log?')) return
  try {
    await fetch('/api/face/clear', { method: 'DELETE' })
    lastVisitResult.value = null
    liveRecognitionResult.value = null
    await refreshAll()
  } catch (err) {
    apiError.value = err.message
  }
}

onMounted(async () => {
  initFaceApi()
  try {
    await fetch('/api/face/import-mc-model', { method: 'POST' })
  } catch {
    // Ignore if backend is offline
  }
  await refreshAll()
  // Automatically attempt opening camera if on camera tab
  if (activeTab.value === 'camera') {
    startCamera()
  }
})

onUnmounted(() => {
  stopCamera()
})
</script>

<template>
  <div class="app-container">
    <!-- Header -->
    <header class="header">
      <div class="header-content">
        <div>
          <h1 class="header-title">
            <span class="badge-accent">Face Recognition (JavaScript)</span>
            Pengenalan Wajah Kamera (Zero-Photo Architecture)
          </h1>
          <p class="header-subtitle">
            Kamera menangkap wajah, mengekstrak <strong>128 float angka (512 bytes)</strong>, lalu <strong>membuang foto</strong>.
            Saat Anda kembali di depan kamera, sistem langsung mengenali wajah Anda berdasarkan kecocokan vektor di database!
          </p>
        </div>
        <div class="header-actions">
          <button
            :class="['btn', isCameraActive ? 'btn-danger' : 'btn-success']"
            @click="isCameraActive ? stopCamera() : startCamera()"
            :disabled="cameraLoading"
          >
            {{ isCameraActive ? '⏹ Matikan Kamera' : '📹 Nyalakan Kamera' }}
          </button>
          <button class="btn btn-primary" :disabled="isLoading" @click="syncMcModel" title="Muat 8 Template Multi-Era Michael Jackson (1969-2000s) ke Database">
            ⚡ Sinkronkan Model Akurat MC (8 Era)
          </button>
          <button class="btn btn-secondary" :disabled="isLoading" @click="refreshAll">
            🔄 Refresh Data
          </button>
          <button class="btn btn-outline-danger" :disabled="isLoading" @click="clearAllData" v-if="profiles.length > 0">
            🗑️ Hapus Semua Data
          </button>
        </div>
      </div>

      <!-- Quick Architecture Alert -->
      <div class="architecture-banner">
        <div class="banner-item red">
          <span class="icon">❌</span>
          <div>
            <strong>Metode Konvensional (Simpan Foto)</strong>
            <p>Simpan file JPG ~2 MB. Berat, membebani database/server, dan privasi rentan jika diretas.</p>
          </div>
        </div>
        <div class="banner-item green">
          <span class="icon">✅</span>
          <div>
            <strong>Metode Sistem Ini (128-D Vector)</strong>
            <p>Hanya 512 Bytes (0.5 KB) per orang. Foto dibuang, privasi aman, pencarian sub-milidetik!</p>
          </div>
        </div>
      </div>

      <!-- Feedback messages -->
      <div v-if="apiSuccess" class="alert alert-success">
        {{ apiSuccess }}
        <button class="close-btn" @click="apiSuccess = null">✕</button>
      </div>
      <div v-if="apiError" class="alert alert-danger">
        {{ apiError }}
        <button class="close-btn" @click="apiError = null">✕</button>
      </div>

      <!-- Navigation Tabs -->
      <nav class="nav-tabs">
        <button
          :class="['tab-btn', { active: activeTab === 'camera' }]"
          @click="activeTab = 'camera'; if (!isCameraActive) startCamera();"
        >
          📹 Kamera Langsung (Webcam Recognition)
        </button>
        <button
          :class="['tab-btn', { active: activeTab === 'simulator' }]"
          @click="activeTab = 'simulator'"
        >
          🎯 Simulasi Virtual Kedatangan
        </button>
        <button
          :class="['tab-btn', { active: activeTab === 'profiles' }]"
          @click="activeTab = 'profiles'"
        >
          👥 Profil Terdaftar ({{ profiles.length }})
        </button>
        <button
          :class="['tab-btn', { active: activeTab === 'storage' }]"
          @click="activeTab = 'storage'"
        >
          📊 Analisis Efisiensi Penyimpanan
        </button>
        <button
          :class="['tab-btn', { active: activeTab === 'logs' }]"
          @click="activeTab = 'logs'"
        >
          📋 Log Riwayat ({{ logs.length }})
        </button>
      </nav>
    </header>

    <!-- TAB 1: LIVE WEBCAM RECOGNITION (UTAMA) -->
    <main v-if="activeTab === 'camera'" class="tab-content">
      <div class="camera-layout">
        <!-- Left: Live Video Feed -->
        <section class="card camera-card">
          <div class="camera-header">
            <div>
              <h2 class="card-title">Live Camera Stream</h2>
              <span class="frame-mode-subtitle">
                Mode Bingkai: <strong>{{ cameraFrameMode === 'ktp' ? '💳 Kartu KTP (Non-Mirror)' : '🧑 Wajah Langsung (Mirror)' }}</strong>
              </span>
            </div>
            <div class="camera-status-pill">
              <span :class="['dot', isCameraActive ? 'green' : 'gray']"></span>
              {{ isCameraActive ? 'Kamera Aktif' : 'Kamera Nonaktif' }}
            </div>
          </div>

          <!-- Switcher Bingkai Kamera (Wajah vs Kartu KTP) -->
          <div class="frame-mode-switcher">
            <button
              type="button"
              :class="['frame-switch-btn', { active: cameraFrameMode === 'ktp' }]"
              @click="cameraFrameMode = 'ktp'; registerSource = 'ktp'; resetPeakLock()"
            >
              💳 Bingkai Kartu KTP (Scan KTP)
            </button>
            <button
              type="button"
              :class="['frame-switch-btn', { active: cameraFrameMode === 'face' }]"
              @click="cameraFrameMode = 'face'; resetPeakLock()"
            >
              🧑 Bingkai Wajah (Tes Wajah Asli)
            </button>
          </div>

          <!-- Switcher Mode Fokus Wajah (Ambil di Frame Aja vs Banyak Orang / Group) -->
          <div class="focus-mode-switcher">
            <button
              type="button"
              :class="['focus-switch-btn', { active: focusMode === 'frame_foreground' }]"
              @click="focusMode = 'frame_foreground'; resetPeakLock()"
            >
              🎯 Ambil di Frame Aja (Muka Paling Depan)
            </button>
            <button
              type="button"
              :class="['focus-switch-btn', { active: focusMode === 'all_group' }]"
              @click="focusMode = 'all_group'; resetPeakLock()"
            >
              👥 Deteksi Banyak Orang (Cari Michael di Group)
            </button>
          </div>

          <!-- Status Bar Lock Persen Paling Besar & Cocok (Kotak Hijau) + Tidak Cocok (Kotak Merah) -->
          <div class="peak-lock-bar">
            <div class="peak-lock-info">
              <span v-if="lockedPeakFace && isPeakLockEnabled" class="peak-lock-badge locked">
                🔒 Kotak Hijau Terkunci (COCOK):
                <strong>{{ lockedPeakFace.top_candidate?.name || 'Michael Jackson' }} ({{ lockedPeakFace.top_candidate?.similarity_percentage ?? 0 }}%)</strong>
              </span>
              <span v-else-if="lockedUnmatchedFace && isPeakLockEnabled" class="peak-lock-badge unmatched">
                🔒 ❌ Kotak Merah (TIDAK COCOK):
                <strong>Bukan {{ lockedUnmatchedFace.top_candidate?.name || 'Target' }} ({{ lockedUnmatchedFace.top_candidate?.similarity_percentage ?? 0 }}% &lt; 70%)</strong>
              </span>
              <span v-else-if="isPeakLockEnabled" class="peak-lock-badge waiting">
                🎯 Auto-Lock Aktif: <strong>Kotak Hijau (Cocok &ge; 70%)</strong> &bull; <strong>Kotak Merah (Tidak Cocok &lt; 70%)</strong>
              </span>
              <span v-else class="peak-lock-badge unlocked">
                🔓 Auto-Lock Nonaktif (Update per Frame)
              </span>
              <span v-if="unmatchedFacesCount > 0 && lockedPeakFace" class="peak-lock-sub-unmatched">
                &bull; ❌ {{ unmatchedFacesCount }} Tidak Cocok (Kotak Merah)
              </span>
            </div>
            <div class="peak-lock-actions">
              <button
                v-if="focusMode === 'all_group' || annotatedPhotoDataUrl"
                type="button"
                :class="['peak-lock-btn', groupBoxDisplayMode === 'all' ? 'active' : '']"
                @click="setGroupBoxDisplayMode(groupBoxDisplayMode === 'all' ? 'highest_only' : 'all')"
              >
                {{ groupBoxDisplayMode === 'all' ? '👥 Semua Kotak (Cocok & Tidak Cocok)' : '🎯 1 Kotak Tertinggi Saja' }}
              </button>
              <button
                v-if="lockedPeakFace || lockedUnmatchedFace"
                type="button"
                class="peak-lock-btn reset"
                @click="resetPeakLock()"
              >
                🔄 Reset Lock
              </button>
              <button
                type="button"
                :class="['peak-lock-btn', isPeakLockEnabled ? 'active' : '']"
                @click="isPeakLockEnabled = !isPeakLockEnabled; if (!isPeakLockEnabled) resetPeakLock()"
              >
                {{ isPeakLockEnabled ? '🔒 Lock Status: ON' : '🔓 Lock Status: OFF' }}
              </button>
            </div>
          </div>

          <!-- Video Wrapper with Reticle Target & Live Bounding Box Canvas -->
          <div class="video-container">
            <video
              v-show="!annotatedPhotoDataUrl"
              ref="videoRef"
              :class="['video-element', { 'unmirrored': cameraFrameMode === 'ktp' || focusMode === 'all_group' }]"
              autoplay
              playsinline
              muted
            ></video>

            <!-- Canvas Pengkotakan Otomatis (Bounding Box Live) -->
            <canvas
              v-show="isCameraActive && !annotatedPhotoDataUrl"
              ref="overlayCanvasRef"
              class="bbox-overlay-canvas"
            ></canvas>

            <!-- Tampilan Hasil Pengkotakan jika Upload Foto / Foto Group -->
            <div v-if="annotatedPhotoDataUrl" class="annotated-photo-wrap">
              <img :src="annotatedPhotoDataUrl" alt="Hasil Pengkotakan Wajah" class="annotated-photo-img" />
              <div class="annotated-photo-bar">
                <div class="annotated-bar-info">
                  <span>🖼️ <strong>{{ recognizedFacesList.length }} Wajah Terdeteksi</strong> ({{ recognizePhotoFileName }})</span>
                  <span class="annotated-sub-note">
                    • ✅ {{ matchedFacesCount }} Cocok (Hijau) • ❌ {{ unmatchedFacesCount }} Tidak Cocok (Merah)
                  </span>
                </div>
                <div class="annotated-bar-actions">
                  <button
                    type="button"
                    :class="['btn btn-xs', groupBoxDisplayMode === 'all' ? 'btn-success' : 'btn-outline-secondary']"
                    @click="setGroupBoxDisplayMode('all')"
                  >
                    👥 Semua Kotak (Cocok & Tidak Cocok)
                  </button>
                  <button
                    type="button"
                    :class="['btn btn-xs', groupBoxDisplayMode === 'highest_only' ? 'btn-success' : 'btn-outline-secondary']"
                    @click="setGroupBoxDisplayMode('highest_only')"
                  >
                    🎯 1 Tertinggi Saja
                  </button>
                  <button type="button" class="btn btn-sm btn-primary" @click="annotatedPhotoDataUrl = null">
                    📹 Kembali ke Live Kamera
                  </button>
                </div>
              </div>
            </div>

            <!-- OVERLAY 1: BINGKAI KARTU SEDERHANA (Kotak Kartu + Kotak Foto Transparan) -->
            <div v-if="isCameraActive && !annotatedPhotoDataUrl && cameraFrameMode === 'ktp'" class="face-target-guide">
              <div class="ktp-card-frame">
                <span class="corner tl"></span>
                <span class="corner tr"></span>
                <span class="corner bl"></span>
                <span class="corner br"></span>

                <!-- Kotak Foto Transparan di Kanan -->
                <div class="ktp-photo-box-guide">
                  <span class="ktp-photo-label">Foto</span>
                </div>

                <span class="target-text">Paskan Kartu di Sini</span>
              </div>
            </div>

            <!-- OVERLAY 2: BINGKAI PANDUAN WAJAH (HANYA DI MODE SATU WAJAH DEPAN) -->
            <div v-if="isCameraActive && !annotatedPhotoDataUrl && cameraFrameMode === 'face' && focusMode === 'frame_foreground'" class="face-target-guide">
              <div class="reticle-box">
                <span class="corner tl"></span>
                <span class="corner tr"></span>
                <span class="corner bl"></span>
                <span class="corner br"></span>
                <span class="target-text">Posisikan Wajah Anda di Sini</span>
              </div>
            </div>

            <!-- Placeholder if camera off -->
            <div v-if="!isCameraActive && !annotatedPhotoDataUrl" class="camera-off-overlay">
              <span class="camera-off-icon">📷</span>
              <p>Kamera belum aktif.</p>
              <button class="btn btn-primary" @click="startCamera" :disabled="cameraLoading">
                {{ cameraLoading ? 'Menghubungkan...' : 'Aktifkan Kamera Sekarang' }}
              </button>
            </div>
          </div>

          <!-- Preview Hasil Crop Terakhir (Hanya di RAM Browser, Tidak Disimpan ke Server) -->
          <div v-if="cameraCaptureThumbnail" class="crop-preview-bar">
            <img :src="cameraCaptureThumbnail" alt="Preview Crop Wajah" class="crop-preview-img" />
            <div class="crop-preview-info">
              <div class="crop-preview-title">
                ✅ Hasil Crop Wajah Terakhir
                <span class="method-pill">{{ extractionMethod || 'ResNet-34 128-D' }}</span>
              </div>
              <p class="crop-preview-desc">
                Preview ini hanya tampil sementara di memori browser agar Anda yakin pas foto KTP / wajah sudah pas di tengah kotak. Server tetap menyimpan <strong>0 foto (hanya 128 angka float)</strong>.
              </p>
            </div>
          </div>

          <!-- Quick Threshold setting -->
          <div class="threshold-control-row">
            <label>Sensitivitas Jarak (Threshold KTP &harr; Wajah): <strong>&lt; {{ thresholdValue.toFixed(2) }}</strong></label>
            <input
              type="range"
              v-model.number="thresholdValue"
              min="0.3"
              max="1.0"
              step="0.02"
              class="slider"
            />
          </div>
        </section>

        <!-- Right: Actions & Results -->
        <div class="camera-actions-col">
          <!-- Action A: DAFTARKAN WAJAH BARU -->
          <section class="card">
            <h2 class="card-title">1. Daftarkan Wajah Baru (KTP / Kamera)</h2>
            <p class="card-desc">
              Pilih sumber pendaftaran: Scan Kartu KTP sesuai bingkai kamera, upload file KTP, atau ambil wajah langsung.
            </p>

            <!-- Mode selector pills -->
            <div class="source-pills-row">
              <button
                type="button"
                :class="['source-pill', { active: registerSource === 'ktp' }]"
                @click="setRegisterSource('ktp')"
              >
                💳 Scan / Upload Kartu KTP
              </button>
              <button
                type="button"
                :class="['source-pill', { active: registerSource === 'camera' }]"
                @click="setRegisterSource('camera')"
              >
                📹 Wajah Kamera Langsung
              </button>
            </div>

            <div class="form-group">
              <div class="form-item">
                <label>Nama Pemilik Wajah / Sesuai KTP:</label>
                <input
                  v-model="cameraRegisterName"
                  type="text"
                  class="form-control"
                  placeholder="Contoh: Budi Santoso"
                />
              </div>

              <div class="form-item">
                <label>Jabatan / Role:</label>
                <select v-model="cameraRegisterRole" class="form-control">
                  <option v-for="r in roles" :key="r" :value="r">{{ r }}</option>
                </select>
              </div>

              <!-- Sub-Mode 1: Camera Face -->
              <div v-if="registerSource === 'camera'">
                <button
                  class="btn btn-primary btn-block btn-lg"
                  :disabled="!isCameraActive || isLoading || !cameraRegisterName.trim()"
                  @click="captureAndRegisterFromCamera"
                >
                  📸 Ambil & Daftarkan Wajah Saya
                </button>
              </div>

              <!-- Sub-Mode 2: KTP Scan (Camera Frame OR File Upload) -->
              <div v-else class="ktp-upload-section">
                <!-- Tombol Scan KTP Langsung dari Bingkai Kamera -->
                <button
                  type="button"
                  class="btn btn-accent-ktp btn-block"
                  :disabled="!isCameraActive || ktpProcessing || isLoading"
                  @click="captureKtpFromCamera"
                >
                  📸 {{ ktpProcessing ? 'Mengekstrak Pas Foto KTP...' : '1. Ambil Pas Foto KTP dari Bingkai Kamera' }}
                </button>

                <div class="or-divider">
                  <span>ATAU UPLOAD FILE FOTO KTP</span>
                </div>

                <div class="ktp-file-dropzone">
                  <input
                    type="file"
                    id="ktpFileInput"
                    accept="image/*"
                    class="file-input-hidden"
                    @change="onKtpFileSelected"
                  />
                  <label for="ktpFileInput" class="file-dropzone-label">
                    <span class="dropzone-icon">💳</span>
                    <span class="dropzone-title">
                      {{ ktpFileName ? ktpFileName : 'Klik untuk Pilih File Foto KTP (JPG/PNG)' }}
                    </span>
                    <span class="dropzone-sub">
                      {{ ktpProcessing ? '⏳ Sedang mendeteksi & meng-crop wajah di KTP...' : (ktpVector ? '✅ Pas Foto KTP Terdeteksi (128 Vektor Siap)' : 'Otomatis mendeteksi & meng-crop kotak foto KTP') }}
                    </span>
                  </label>
                </div>

                <!-- Preview Pas Foto KTP yang berhasil di-crop -->
                <div v-if="ktpPreviewThumbnail" class="ktp-cropped-preview-box">
                  <img :src="ktpPreviewThumbnail" alt="Pas Foto KTP" class="ktp-cropped-img" />
                  <div class="ktp-cropped-meta">
                    <strong>✅ Pas Foto KTP Siap Didaftarkan</strong>
                    <span>Metode: {{ extractionMethod }}</span>
                    <small>Pastikan gambar di samping tepat menampilkan wajah pada KTP Anda.</small>
                  </div>
                </div>

                <button
                  class="btn btn-primary btn-block btn-lg"
                  :disabled="!ktpVector || isLoading || !cameraRegisterName.trim()"
                  @click="registerFromKtp"
                >
                  📥 2. Simpan Vektor Wajah KTP ke Database
                </button>
              </div>
            </div>

            <div class="security-note">
              🔒 <strong>Prinsip Zero-Photo:</strong> Foto KTP maupun foto kamera asli <strong>langsung dibuang permanen</strong> dari memori. Yang disimpan di database MySQL hanyalah 128 angka float (512 bytes).
            </div>
          </section>

          <!-- Action B: KENALI WAJAH SAYA (VERIFIKASI & PENGKOTAKAN GROUP) -->
          <section class="card">
            <h2 class="card-title">2. Kenali & Bandingkan dengan Personil</h2>
            <p class="card-desc">
              Sistem otomatis melakukan <strong>Pengkotakan Wajah (Bounding Box)</strong> pada semua wajah di kamera maupun foto upload, lalu membandingkannya dengan seluruh personil di database.
            </p>

            <div class="action-buttons-stack">
              <button
                class="btn btn-success btn-block btn-lg"
                :disabled="!isCameraActive || isLoading || profiles.length === 0"
                @click="captureAndRecognizeFromCamera('face')"
              >
                🧑 Kenali & Kotakkan Wajah di Kamera (Single / Group)
              </button>

              <!-- Upload Foto / Foto Group untuk Tes Pengkotakan & Perbandingan Personil -->
              <div class="recognize-photo-upload-wrap">
                <input
                  type="file"
                  id="recognizePhotoInput"
                  accept="image/*"
                  class="file-input-hidden"
                  :disabled="isLoading || profiles.length === 0"
                  @change="onRecognizePhotoSelected"
                />
                <label
                  for="recognizePhotoInput"
                  :class="['btn btn-accent-group btn-block', { disabled: isLoading || profiles.length === 0 }]"
                >
                  🖼️ Upload Foto / Group untuk Pengkotakan & Bandingkan Personil
                </label>
              </div>

              <button
                class="btn btn-outline-info btn-block"
                :disabled="!isCameraActive || isLoading || profiles.length === 0"
                @click="captureAndRecognizeFromCamera('ktp_camera')"
              >
                💳 Tes Cocokkan Kartu KTP di Kamera (Mode Bingkai KTP)
              </button>

              <button
                :class="['btn btn-block', isAutoScan ? 'btn-danger' : 'btn-secondary']"
                :disabled="!isCameraActive || profiles.length === 0"
                @click="toggleAutoScan"
              >
                {{ isAutoScan ? '⏹ Matikan Auto-Scan' : '⚡ Mode Kiosk (Auto-Scan Tiap 2.5 Detik)' }}
              </button>
            </div>

            <div v-if="profiles.length === 0" class="callout callout-warning">
              ⚠️ Belum ada profil terdaftar di database. Silakan daftarkan KTP/wajah Anda di bagian atas terlebih dahulu.
            </div>
          </section>
        </div>
      </div>

      <!-- Live Recognition Result Box (Muncul di Bawah) -->
      <section v-if="liveRecognitionResult" class="card result-card">
        <div class="result-header-row">
          <h2 class="card-title">
            Hasil Pengkotakan Wajah & Perbandingan Personil ({{ recognizedFacesList.length }} Wajah Terdeteksi)
          </h2>
          <span class="badge badge-info">Zero-Photo Vector Comparison</span>
        </div>

        <!-- TARGET SUMMARY BANNER (MENUNJUKKAN LANGSUNG POSISI TARGET MICHAEL DI GROUP / FRAME) -->
        <div
          v-if="liveRecognitionResult.target_summary"
          :class="['target-pinpoint-banner', liveRecognitionResult.target_detected ? 'is-detected' : 'is-not-detected']"
        >
          <div class="target-pinpoint-icon">
            {{ liveRecognitionResult.target_detected ? '🎯' : '🔍' }}
          </div>
          <div class="target-pinpoint-text">
            <h4 class="target-pinpoint-heading">
              {{ liveRecognitionResult.target_detected ? 'Target Teridentifikasi di Group!' : 'Status Pelacakan Target' }}
            </h4>
            <p class="target-pinpoint-msg">{{ liveRecognitionResult.target_summary }}</p>
          </div>
          <div v-if="liveRecognitionResult.target_face_index" class="target-pinpoint-badge">
            🌟 Kotak #{{ liveRecognitionResult.target_face_index }} Adalah Michael
          </div>
        </div>

        <!-- KARTU PENGKOTAKAN PER WAJAH (MENDUKUNG 1 WAJAH MAUPUN MULTI-FACE / GROUP) -->
        <div v-if="sortedRecognizedFaces.length > 0" class="detected-faces-grid">
          <div
            v-for="faceItem in sortedRecognizedFaces"
            :key="faceItem.face_index"
            :class="[
              'face-comparison-card',
              {
                'face-is-target': isWinningTargetFace(faceItem) && isFacePassed(faceItem),
                'face-is-unmatched': !isFacePassed(faceItem),
              },
            ]"
          >
            <div class="face-card-top">
              <div class="face-thumb-box">
                <img
                  v-if="faceItem.previewThumbnail"
                  :src="faceItem.previewThumbnail"
                  :alt="'Wajah #' + faceItem.face_index"
                  class="face-thumb-img"
                  :style="{
                    borderColor: isWinningTargetFace(faceItem) && isFacePassed(faceItem)
                      ? '#10b981'
                      : !isFacePassed(faceItem)
                        ? '#ef4444'
                        : '#f59e0b',
                  }"
                />
                <span
                  class="face-index-tag"
                  :style="{
                    backgroundColor: isWinningTargetFace(faceItem) && isFacePassed(faceItem)
                      ? '#10b981'
                      : !isFacePassed(faceItem)
                        ? '#ef4444'
                        : '#f59e0b',
                  }"
                >
                  Kotak #{{ faceItem.face_index }}
                </span>
                <span
                  v-if="isWinningTargetFace(faceItem) && isFacePassed(faceItem) && isMichaelTarget(faceItem)"
                  class="face-target-chip"
                >
                  🔒 INI MICHAEL
                </span>
                <span
                  v-else-if="isWinningTargetFace(faceItem) && isFacePassed(faceItem)"
                  class="face-target-chip"
                >
                  🔒 COCOK ({{ faceItem.top_candidate?.similarity_percentage || 0 }}%)
                </span>
                <span
                  v-else-if="!isFacePassed(faceItem)"
                  class="face-unmatched-chip"
                >
                  ❌ TIDAK COCOK ({{ faceItem.top_candidate?.similarity_percentage || 0 }}%)
                </span>
              </div>

              <div class="face-match-summary">
                <span class="compared-label">Status & Perbandingan Personil:</span>
                <div class="personil-winner-row">
                  <strong class="personil-winner-name">
                    {{ faceItem.top_candidate?.name || 'Tidak Diketahui' }}
                  </strong>
                  <span
                    class="personil-winner-pct"
                    :style="{
                      backgroundColor: isWinningTargetFace(faceItem) && isFacePassed(faceItem)
                        ? '#10b981'
                        : !isFacePassed(faceItem)
                          ? '#ef4444'
                          : '#f59e0b',
                    }"
                  >
                    {{ isFacePassed(faceItem) ? (faceItem.top_candidate?.similarity_percentage || 0) + '% Cocok' : (faceItem.top_candidate?.similarity_percentage || 0) + '% Tidak Cocok' }}
                  </span>
                </div>
                <div
                  v-if="isWinningTargetFace(faceItem) && isFacePassed(faceItem) && isMichaelTarget(faceItem)"
                  class="michael-winner-banner"
                >
                  🔒 KOTAK HIJAU (COCOK TERTINGGI): Wajah pada Kotak #{{ faceItem.face_index }} adalah Michael Jackson ({{ faceItem.top_candidate?.similarity_percentage }}%)!
                </div>
                <div
                  v-else-if="isWinningTargetFace(faceItem) && isFacePassed(faceItem)"
                  class="michael-winner-banner"
                >
                  🔒 KOTAK HIJAU (COCOK TERTINGGI): Wajah ini cocok dengan {{ faceItem.top_candidate?.name }} ({{ faceItem.top_candidate?.similarity_percentage }}%)
                </div>
                <div
                  v-else-if="!isFacePassed(faceItem)"
                  class="michael-unmatched-banner"
                >
                  ❌ KOTAK MERAH (TIDAK COCOK): Wajah pada Kotak #{{ faceItem.face_index }} bukan {{ faceItem.top_candidate?.name || 'Target' }} (Kemiripan {{ faceItem.top_candidate?.similarity_percentage || 0 }}% di bawah ambang 70%)
                </div>
                <div
                  v-else
                  class="michael-other-banner"
                >
                  Kandidat Lain di Group ({{ faceItem.top_candidate?.similarity_percentage || 0 }}%)
                </div>
                <span class="personil-role-sub">
                  Role: {{ faceItem.top_candidate?.role || '-' }} • ID: {{ faceItem.top_candidate?.identifier || '-' }} • Jarak L2: {{ faceItem.metrics?.euclidean_distance }}
                </span>
                <span v-if="faceItem.second_candidate" class="runner-up-note">
                  Pembanding Ke-2: <strong>{{ faceItem.second_candidate.name }}</strong> ({{ faceItem.second_candidate.similarity_percentage }}% • Selisih Jarak: {{ faceItem.metrics?.margin_gap }})
                </span>
              </div>
            </div>

            <!-- Bar Perbandingan Wajah Ini terhadap Semua Personil Terdaftar -->
            <div class="personil-bars-section">
              <span class="bars-title">Dibandingkan dengan Semua Personil di Database:</span>
              <div class="personil-bars-list">
                <div
                  v-for="(cand, cIdx) in faceItem.candidates"
                  :key="cand.id"
                  :class="['personil-bar-item', { 'is-top': cIdx === 0 }]"
                >
                  <div class="personil-bar-header">
                    <span class="personil-cand-name">
                      <span class="rank-num">#{{ cIdx + 1 }}</span>
                      <strong>{{ cand.name }}</strong>
                      <small>({{ cand.role }})</small>
                      <small v-if="cand.matched_era && cand.matched_era !== 'Centroid Utama'" style="color: #10b981; margin-left: 6px; font-weight: 600;">
                        • {{ cand.matched_era }}
                      </small>
                    </span>
                    <span class="personil-cand-Right">
                      <small class="dist-tag">L2: {{ cand.distance }}</small>
                      <strong :style="{ color: percentageColor(cand.similarity_percentage || cand.confidence) }">
                        {{ cand.similarity_percentage || cand.confidence }}%
                      </strong>
                    </span>
                  </div>
                  <div class="progress-bar-bg">
                    <div
                      class="progress-bar-fill"
                      :style="{
                        width: (cand.similarity_percentage || cand.confidence) + '%',
                        backgroundColor: percentageColor(cand.similarity_percentage || cand.confidence)
                      }"
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Meteran Persentase Kemiripan Utama -->
        <div class="percentage-gauge-card">
          <div class="gauge-circle-wrap">
            <svg viewBox="0 0 120 120" class="gauge-svg">
              <circle cx="60" cy="60" r="48" class="gauge-track" />
              <circle
                cx="60" cy="60" r="48"
                class="gauge-progress"
                :stroke="percentageColor(liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage)"
                :stroke-dasharray="301.6"
                :stroke-dashoffset="301.6 * (1 - (liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage) / 100)"
              />
            </svg>
            <div class="gauge-center-content">
              <span class="gauge-big-num" :style="{ color: percentageColor(liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage) }">
                {{ liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage }}%
              </span>
              <span class="gauge-mini-label">Kemiripan</span>
            </div>
          </div>

          <div class="gauge-details">
            <div class="gauge-status-title" :style="{ color: percentageColor(liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage) }">
              {{ similarityStatusText(liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage, liveRecognitionResult.is_match) }}
            </div>
            <p v-if="liveRecognitionResult.top_candidate" class="gauge-detail-desc">
              Hasil komparasi vektor: Wajah di kamera terdeteksi <strong>{{ liveRecognitionResult.top_candidate.similarity_percentage }}% mirip</strong> dengan <strong>{{ liveRecognitionResult.top_candidate.name }}</strong> ({{ liveRecognitionResult.top_candidate.role }}).
            </p>
            <div class="gauge-legend">
              <span class="legend-badge red">&lt; 50% Beda Orang</span>
              <span class="legend-badge yellow">50% - 69% Mirip Sedang</span>
              <span class="legend-badge green">&ge; 70% Wajah Cocok (Lolos)</span>
            </div>
          </div>
        </div>

        <div
          :class="[
            'verdict-badge',
            liveRecognitionResult.is_match ? 'match' : 'no-match'
          ]"
        >
          <div class="verdict-header">
            <span class="verdict-icon">
              {{ liveRecognitionResult.is_match ? '✅ WAJAH DIKENALI!' : '🚫 WAJAH TIDAK COCOK / TIDAK TERDAFTAR' }}
            </span>
            <span class="badge-time">Baru saja</span>
          </div>

          <h3 v-if="liveRecognitionResult.is_match && liveRecognitionResult.matched_profile" class="matched-name-title">
            Halo, {{ liveRecognitionResult.matched_profile.name }}!
            <span class="matched-role">({{ liveRecognitionResult.matched_profile.role }} - {{ liveRecognitionResult.matched_profile.identifier }})</span>
          </h3>
          <p class="verdict-message">
            {{ liveRecognitionResult.message }}
          </p>

          <!-- Top candidate percentage highlight -->
          <div v-if="liveRecognitionResult.top_candidate" class="top-candidate-banner">
            <div class="top-candidate-box">
              <span class="top-candidate-text">
                Wajah di kamera paling mirip dengan:
                <strong>{{ liveRecognitionResult.top_candidate.name }} ({{ liveRecognitionResult.top_candidate.role }})</strong>
              </span>
              <span :class="['top-percentage-badge', liveRecognitionResult.is_match ? 'match' : 'no-match']">
                {{ liveRecognitionResult.top_candidate.similarity_percentage }}% Kemiripan
              </span>
            </div>
          </div>
        </div>

        <!-- Metrics Grid -->
        <div class="metrics-grid">
          <div class="metric-card">
            <span class="metric-title">Tingkat Kemiripan (%)</span>
            <span class="metric-val highlight">
              {{ liveRecognitionResult.metrics.similarity_percentage || liveRecognitionResult.metrics.confidence_percentage }}%
            </span>
            <span class="metric-note">Standar Lolos: &ge; 70%</span>
          </div>

          <div class="metric-card">
            <span class="metric-title">Jarak Euclidean (L2)</span>
            <span class="metric-val">
              {{ liveRecognitionResult.metrics.euclidean_distance }}
            </span>
            <span class="metric-note">
              Ambang batas: &lt; {{ liveRecognitionResult.metrics.threshold_used }}
            </span>
          </div>

          <div class="metric-card">
            <span class="metric-title">Cosine Similarity</span>
            <span class="metric-val">
              {{ liveRecognitionResult.metrics.cosine_similarity }}
            </span>
            <span class="metric-note">Maksimum: 1.0000</span>
          </div>
        </div>

        <!-- Candidates Match Ranking with Progress Bars -->
        <div class="candidates-table-wrap">
          <h4>Peringkat Kemiripan dengan Semua Orang di Database:</h4>
          <table class="data-table">
            <thead>
              <tr>
                <th>Nama Profil</th>
                <th>ID</th>
                <th>Tingkat Kemiripan (%)</th>
                <th>Jarak Euclidean</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="cand in liveRecognitionResult.candidates"
                :key="cand.id"
                :class="{ 'row-winner': cand.id === liveRecognitionResult.matched_profile?.id }"
              >
                <td><strong>{{ cand.name }}</strong></td>
                <td>{{ cand.identifier }}</td>
                <td>
                  <div class="similarity-progress-wrap">
                    <div class="progress-bar-bg">
                      <div
                        class="progress-bar-fill"
                        :style="{
                          width: (cand.similarity_percentage || cand.confidence) + '%',
                          backgroundColor: (cand.similarity_percentage || cand.confidence) >= 70 ? '#10b981' : ((cand.similarity_percentage || cand.confidence) >= 50 ? '#f59e0b' : '#ef4444')
                        }"
                      ></div>
                    </div>
                    <span class="percentage-text">{{ cand.similarity_percentage || cand.confidence }}%</span>
                  </div>
                </td>
                <td>{{ cand.distance }}</td>
                <td>
                  <span
                    v-if="(cand.similarity_percentage || cand.confidence) >= 70"
                    class="badge badge-success"
                  >
                    Cocok (Match)
                  </span>
                  <span
                    v-else-if="(cand.similarity_percentage || cand.confidence) >= 50"
                    class="badge badge-warning"
                  >
                    Mirip Sedang
                  </span>
                  <span v-else class="badge badge-danger">
                    Beda Orang
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Extracted sample vector for transparency -->
        <div class="vector-preview-box">
          <span class="vector-title">Vektor Fitur Wajah Kamera (8 dari 128 Angka Float):</span>
          <code>
            [ {{ lastExtractedSample.join(', ') }} ... ]
          </code>
        </div>
      </section>
    </main>

    <!-- TAB 2: VIRTUAL SIMULATOR -->
    <main v-if="activeTab === 'simulator'" class="tab-content">
      <div class="grid-2col">
        <!-- Control Panel -->
        <section class="card">
          <div class="card-header-flex">
            <div>
              <h2 class="card-title">Simulasi Virtual Kedatangan</h2>
              <p class="card-desc">Uji coba kedatangan dengan variasi sudut miring, cahaya redup, kacamata, atau orang asing.</p>
            </div>
            <button class="btn btn-primary" :disabled="isLoading" @click="seedFiveFaces">
              ✨ Isi 5 Profil Contoh
            </button>
          </div>

          <div v-if="profiles.length === 0" class="empty-state">
            <p>Belum ada profil di sistem.</p>
            <button class="btn btn-primary" @click="seedFiveFaces">
              ✨ Klik untuk generate 5 profil contoh
            </button>
          </div>

          <form v-else @submit.prevent="runVisitSimulation" class="form-group">
            <div class="form-item">
              <label>Pilih Orang yang Datang:</label>
              <select
                v-model="selectedProfileId"
                :disabled="selectedCondition === 'unknown_stranger'"
                class="form-control"
              >
                <option v-for="p in profiles" :key="p.id" :value="p.id">
                  {{ p.name }} ({{ p.identifier }} - {{ p.role }})
                </option>
              </select>
            </div>

            <div class="form-item">
              <label>Kondisi Saat Kedatangan Berikutnya:</label>
              <select v-model="selectedCondition" class="form-control">
                <option v-for="c in conditions" :key="c.key" :value="c.key">
                  {{ c.label }}
                </option>
              </select>
            </div>

            <div class="form-item">
              <div class="slider-label">
                <label>Ambang Batas Jarak (Threshold):</label>
                <span class="threshold-badge">{{ thresholdValue.toFixed(2) }}</span>
              </div>
              <input
                type="range"
                v-model.number="thresholdValue"
                min="0.2"
                max="1.0"
                step="0.05"
                class="slider"
              />
            </div>

            <button type="submit" class="btn btn-success btn-block" :disabled="isLoading">
              🚀 Simulasikan & Cocokkan Wajah
            </button>
          </form>
        </section>

        <!-- Result Panel -->
        <section class="card">
          <h2 class="card-title">Hasil Pengenalan Simulasi</h2>

          <div v-if="!lastVisitResult" class="placeholder-box">
            <span class="placeholder-icon">🎯</span>
            <p>Pilih profil & kondisi kedatangan di samping, lalu klik <strong>"Simulasikan & Cocokkan Wajah"</strong>.</p>
          </div>

          <div v-else class="result-details">
            <!-- Meteran Persentase Kemiripan Simulasi -->
            <div class="percentage-gauge-card">
              <div class="gauge-circle-wrap">
                <svg viewBox="0 0 120 120" class="gauge-svg">
                  <circle cx="60" cy="60" r="48" class="gauge-track" />
                  <circle
                    cx="60" cy="60" r="48"
                    class="gauge-progress"
                    :stroke="percentageColor(lastVisitResult.recognition_result.metrics.similarity_percentage || lastVisitResult.recognition_result.metrics.confidence_percentage)"
                    :stroke-dasharray="301.6"
                    :stroke-dashoffset="301.6 * (1 - (lastVisitResult.recognition_result.metrics.similarity_percentage || lastVisitResult.recognition_result.metrics.confidence_percentage) / 100)"
                  />
                </svg>
                <div class="gauge-center-content">
                  <span class="gauge-big-num" :style="{ color: percentageColor(lastVisitResult.recognition_result.metrics.similarity_percentage || lastVisitResult.recognition_result.metrics.confidence_percentage) }">
                    {{ lastVisitResult.recognition_result.metrics.similarity_percentage || lastVisitResult.recognition_result.metrics.confidence_percentage }}%
                  </span>
                  <span class="gauge-mini-label">Kemiripan</span>
                </div>
              </div>

              <div class="gauge-details">
                <div class="gauge-status-title" :style="{ color: percentageColor(lastVisitResult.recognition_result.metrics.similarity_percentage || lastVisitResult.recognition_result.metrics.confidence_percentage) }">
                  {{ similarityStatusText(lastVisitResult.recognition_result.metrics.similarity_percentage || lastVisitResult.recognition_result.metrics.confidence_percentage, lastVisitResult.recognition_result.is_match) }}
                </div>
                <p v-if="lastVisitResult.recognition_result.top_candidate" class="gauge-detail-desc">
                  Hasil simulasi: Terdeteksi <strong>{{ lastVisitResult.recognition_result.top_candidate.similarity_percentage }}% mirip</strong> dengan <strong>{{ lastVisitResult.recognition_result.top_candidate.name }}</strong> ({{ lastVisitResult.recognition_result.top_candidate.role }}).
                </p>
                <div class="gauge-legend">
                  <span class="legend-badge red">&lt; 50% Beda Orang</span>
                  <span class="legend-badge yellow">50% - 69% Mirip Sedang</span>
                  <span class="legend-badge green">&ge; 70% Wajah Cocok (Lolos)</span>
                </div>
              </div>
            </div>

            <div
              :class="[
                'verdict-badge',
                lastVisitResult.recognition_result.is_match ? 'match' : 'no-match'
              ]"
            >
              <span class="verdict-icon">
                {{ lastVisitResult.recognition_result.is_match ? '✅ MATCH DITEMUKAN' : '🚫 WAJAH TIDAK COCOK / DITOLAK' }}
              </span>
              <p class="verdict-message">
                {{ lastVisitResult.recognition_result.message }}
              </p>
            </div>

            <div class="metrics-grid">
              <div class="metric-card">
                <span class="metric-title">Jarak Euclidean</span>
                <span class="metric-val highlight">
                  {{ lastVisitResult.recognition_result.metrics.euclidean_distance }}
                </span>
                <span class="metric-note">&lt; {{ lastVisitResult.recognition_result.metrics.threshold_used }}</span>
              </div>

              <div class="metric-card">
                <span class="metric-title">Cosine Similarity</span>
                <span class="metric-val">
                  {{ lastVisitResult.recognition_result.metrics.cosine_similarity }}
                </span>
                <span class="metric-note">Maks: 1.0000</span>
              </div>

              <div class="metric-card">
                <span class="metric-title">Confidence</span>
                <span class="metric-val">
                  {{ lastVisitResult.recognition_result.metrics.confidence_percentage }}%
                </span>
                <span class="metric-note">Akurasi</span>
              </div>
            </div>

            <div class="candidates-table-wrap">
              <h4>Perbandingan Kemiripan dengan Semua Profil:</h4>
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Nama Profil</th>
                    <th>Tingkat Kemiripan (%)</th>
                    <th>Jarak Euclidean</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="cand in lastVisitResult.recognition_result.candidates"
                    :key="cand.id"
                    :class="{ 'row-winner': cand.id === lastVisitResult.recognition_result.matched_profile?.id }"
                  >
                    <td><strong>{{ cand.name }}</strong></td>
                    <td>
                      <div class="similarity-progress-wrap">
                        <div class="progress-bar-bg">
                          <div
                            class="progress-bar-fill"
                            :style="{
                              width: (cand.similarity_percentage || cand.confidence) + '%',
                              backgroundColor: (cand.similarity_percentage || cand.confidence) >= 70 ? '#10b981' : ((cand.similarity_percentage || cand.confidence) >= 50 ? '#f59e0b' : '#ef4444')
                            }"
                          ></div>
                        </div>
                        <span class="percentage-text">{{ cand.similarity_percentage || cand.confidence }}%</span>
                      </div>
                    </td>
                    <td>{{ cand.distance }}</td>
                    <td>
                      <span
                        v-if="(cand.similarity_percentage || cand.confidence) >= 70"
                        class="badge badge-success"
                      >
                        Match (Cocok)
                      </span>
                      <span
                        v-else-if="(cand.similarity_percentage || cand.confidence) >= 50"
                        class="badge badge-warning"
                      >
                        Mirip Sedang
                      </span>
                      <span v-else class="badge badge-danger">
                        Beda Orang
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </main>

    <!-- TAB 3: PROFILES -->
    <main v-if="activeTab === 'profiles'" class="tab-content">
      <div class="grid-2col">
        <section class="card">
          <div class="card-header-flex">
            <div>
              <h2 class="card-title">Daftar Profil Wajah Terdaftar</h2>
              <p class="card-desc">Semua profil HANYA menyimpan 128 float angka (512 bytes), tanpa ada 1 foto pun di server.</p>
            </div>
            <button class="btn btn-outline-danger" @click="clearAllData" v-if="profiles.length > 0">
              🗑️ Reset Semua
            </button>
          </div>

          <div v-if="profiles.length === 0" class="empty-state">
            <p>Database kosong. Daftarkan lewat kamera atau buat simulasi.</p>
            <button class="btn btn-primary" @click="seedFiveFaces">
              ✨ Generate 5 Profil Otomatis
            </button>
          </div>

          <div v-else class="profiles-list">
            <div v-for="p in profiles" :key="p.id" class="profile-item">
              <div class="avatar-circle sm">
                {{ p.name.charAt(0) }}
              </div>
              <div class="profile-details">
                <div class="profile-title-row">
                  <h3>{{ p.name }}</h3>
                  <span class="badge badge-info">{{ p.identifier }}</span>
                </div>
                <p class="profile-meta">
                  <span>Jabatan: {{ p.role }}</span> •
                  <span>Gender: {{ p.gender }}</span> •
                  <span>Dimensi Vektor: {{ p.dimensions }}</span>
                </p>
                <div class="storage-tag">
                  💾 Ukuran di DB: <strong>{{ p.storage_bytes }} Bytes (0.51 KB)</strong> | Foto: <strong>0 KB (Tanpa Foto)</strong>
                </div>
                <div class="vector-snippet">
                  Vektor float preview: [{{ p.embedding_preview.join(', ') }}...]
                </div>
              </div>
              <div class="profile-actions">
                <button class="btn-icon danger" @click="deleteProfile(p.id)" title="Hapus profil">
                  🗑️
                </button>
              </div>
            </div>
          </div>
        </section>

        <!-- Manual Profile Form -->
        <section class="card">
          <h2 class="card-title">Tambah Profil Manual</h2>
          <p class="card-desc">
            Daftarkan profil tanpa kamera (menggunakan simulasi ekstraksi canonical vector).
          </p>

          <form @submit.prevent="registerProfile" class="form-group">
            <div class="form-item">
              <label>Nama Lengkap:</label>
              <input v-model="newName" type="text" class="form-control" placeholder="Contoh: Andi Pratama" required />
            </div>

            <div class="form-item">
              <label>Kode / ID:</label>
              <input v-model="newIdentifier" type="text" class="form-control" placeholder="Contoh: EMP-099" />
            </div>

            <div class="form-item">
              <label>Jabatan:</label>
              <select v-model="newRole" class="form-control">
                <option v-for="r in roles" :key="r" :value="r">{{ r }}</option>
              </select>
            </div>

            <div class="form-item">
              <label>Gender:</label>
              <select v-model="newGender" class="form-control">
                <option value="Male">Pria (Male)</option>
                <option value="Female">Wanita (Female)</option>
              </select>
            </div>

            <button type="submit" class="btn btn-primary btn-block" :disabled="isLoading">
              ➕ Simpan Profil
            </button>
          </form>
        </section>
      </div>
    </main>

    <!-- TAB 4: STORAGE ANALYTICS -->
    <main v-if="activeTab === 'storage'" class="tab-content">
      <section class="card" v-if="storageMetrics">
        <h2 class="card-title">Perbandingan Efisiensi: Foto Mentah vs 128-D Vector</h2>
        <p class="card-desc">
          Bukti efisiensi penyimpanan: Kenapa 128-D vector adalah standar industri keamanan dan perbankan modern.
        </p>

        <div class="stats-row">
          <div class="stat-card danger">
            <span class="stat-label">Jika Simpan Foto Mentah (JPG/PNG)</span>
            <span class="stat-value">{{ storageMetrics.per_user.raw_photo_formatted }}</span>
            <span class="stat-sub">Per orang / pendaftaran</span>
          </div>

          <div class="stat-card success">
            <span class="stat-label">Metode Sistem Kita (128-D Float Vector)</span>
            <span class="stat-value">{{ storageMetrics.per_user.vector_binary_formatted }}</span>
            <span class="stat-sub">512 Bytes (Binary) / ~0.93 KB (JSON)</span>
          </div>

          <div class="stat-card highlight">
            <span class="stat-label">Persentase Penghematan Ruang</span>
            <span class="stat-value">{{ storageMetrics.per_user.savings_percentage }}%</span>
            <span class="stat-sub">Lebih hemat &gt; 4.000 kali lipat!</span>
          </div>
        </div>

        <h3 class="subsection-title">Simulasi Skala Pengguna:</h3>
        <table class="data-table">
          <thead>
            <tr>
              <th>Jumlah Pengguna</th>
              <th>Total Storage (Foto Mentah)</th>
              <th>Total Storage (Vector Embeddings)</th>
              <th>Efisiensi Penghematan</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="row in storageMetrics.scale_simulation" :key="row.users">
              <td><strong>{{ row.users.toLocaleString() }} Pengguna</strong></td>
              <td class="text-danger">{{ row.photo_storage }}</td>
              <td class="text-success"><strong>{{ row.vector_storage }}</strong></td>
              <td><span class="badge badge-success">{{ row.savings }}</span></td>
            </tr>
          </tbody>
        </table>
      </section>
    </main>

    <!-- TAB 5: AUDIT LOGS -->
    <main v-if="activeTab === 'logs'" class="tab-content">
      <section class="card">
        <h2 class="card-title">Riwayat Log Kedatangan & Verifikasi Wajah</h2>
        <p class="card-desc">
          Audit trail setiap pengenalan wajah: jarak Euclidean, confidence, dan status kecocokan.
        </p>

        <div v-if="logs.length === 0" class="empty-state">
          <p>Belum ada riwayat verifikasi.</p>
        </div>

        <table v-else class="data-table">
          <thead>
            <tr>
              <th>Waktu</th>
              <th>Hasil Pengenalan</th>
              <th>Status</th>
              <th>Jarak Euclidean</th>
              <th>Cosine Sim</th>
              <th>Confidence</th>
              <th>Kondisi / Sumber</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="log in logs" :key="log.id">
              <td>{{ log.time_ago }}</td>
              <td><strong>{{ log.recognized_name }}</strong></td>
              <td>
                <span :class="['badge', log.is_match ? 'badge-success' : 'badge-danger']">
                  {{ log.is_match ? 'MATCH' : 'TIDAK COCOK' }}
                </span>
              </td>
              <td>{{ log.distance }}</td>
              <td>{{ log.similarity }}</td>
              <td>{{ log.confidence }}%</td>
              <td><small>{{ log.condition }}</small></td>
            </tr>
          </tbody>
        </table>
      </section>
    </main>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.header {
  display: flex;
  flex-direction: column;
  gap: 1.25rem;
  background: var(--bg-card);
  padding: 1.5rem;
  border-radius: 12px;
  border: 1px solid var(--border);
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  flex-wrap: wrap;
  gap: 1rem;
}

.header-title {
  font-size: 1.5rem;
  font-weight: 700;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.header-subtitle {
  color: var(--text-muted);
  font-size: 0.95rem;
  max-width: 750px;
  margin-top: 0.5rem;
}

.badge-accent {
  align-self: flex-start;
  font-size: 0.75rem;
  background: #2563eb;
  color: #fff;
  padding: 0.2rem 0.6rem;
  border-radius: 9999px;
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.05em;
}

.header-actions {
  display: flex;
  gap: 0.75rem;
}

/* Architecture banner */
.architecture-banner {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

@media (max-width: 768px) {
  .architecture-banner {
    grid-template-columns: 1fr;
  }
}

.banner-item {
  display: flex;
  gap: 0.75rem;
  padding: 0.85rem 1rem;
  border-radius: 8px;
  font-size: 0.88rem;
  align-items: flex-start;
}

.banner-item.red {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.banner-item.green {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.banner-item .icon {
  font-size: 1.25rem;
}

.banner-item p {
  color: var(--text-muted);
  font-size: 0.82rem;
  margin-top: 0.2rem;
}

/* Navigation Tabs */
.nav-tabs {
  display: flex;
  gap: 0.5rem;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
  overflow-x: auto;
}

.tab-btn {
  background: transparent;
  color: var(--text-muted);
  border: none;
  padding: 0.6rem 1rem;
  font-size: 0.9rem;
  font-weight: 600;
  border-radius: 6px;
  transition: all 0.2s ease;
  white-space: nowrap;
}

.tab-btn:hover {
  background: var(--bg-card-subtle);
  color: #fff;
}

.tab-btn.active {
  background: #3b82f6;
  color: #fff;
}

/* Camera Layout */
.camera-layout {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 1.5rem;
}

@media (max-width: 950px) {
  .camera-layout {
    grid-template-columns: 1fr;
  }
}

.camera-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.camera-status-pill {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.8rem;
  font-weight: 600;
  background: var(--bg-main);
  padding: 0.3rem 0.7rem;
  border-radius: 9999px;
  border: 1px solid var(--border);
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.dot.green {
  background: #10b981;
  box-shadow: 0 0 8px #10b981;
}
.dot.gray {
  background: #64748b;
}

/* Peak Lock Status Bar */
.peak-lock-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  margin-bottom: 0.65rem;
  padding: 0.45rem 0.7rem;
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid rgba(16, 185, 129, 0.35);
  border-radius: 8px;
  font-size: 0.78rem;
}

.peak-lock-info {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.peak-lock-badge {
  display: inline-flex;
  align-items: center;
  gap: 0.3rem;
  line-height: 1.3;
}

.peak-lock-badge.locked {
  color: #34d399;
}

.peak-lock-badge.unmatched {
  color: #f87171;
}

.peak-lock-sub-unmatched {
  color: #f87171;
  font-weight: 600;
  font-size: 0.74rem;
}

.peak-lock-badge.waiting {
  color: #94a3b8;
}

.peak-lock-badge.unlocked {
  color: #fbbf24;
}

.peak-lock-actions {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.peak-lock-btn {
  padding: 0.25rem 0.6rem;
  border-radius: 6px;
  border: 1px solid #334155;
  background: #1e293b;
  color: #cbd5e1;
  font-size: 0.74rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}

.peak-lock-btn.active {
  background: rgba(16, 185, 129, 0.18);
  border-color: #10b981;
  color: #34d399;
}

.peak-lock-btn.reset {
  background: rgba(245, 158, 11, 0.16);
  border-color: #f59e0b;
  color: #fbbf24;
}

.peak-lock-btn:hover {
  filter: brightness(1.15);
}

/* Video element & Reticle */
.video-container {
  position: relative;
  width: 100%;
  aspect-ratio: 4 / 3;
  background: #000;
  border-radius: 10px;
  overflow: hidden;
  border: 2px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: center;
}

.video-element {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}

.video-element.unmirrored {
  transform: none;
}

.bbox-overlay-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 5;
}

.annotated-photo-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #0f172a;
}

.annotated-photo-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.annotated-photo-bar {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(15, 23, 42, 0.95);
  padding: 0.5rem 0.85rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
  font-size: 0.78rem;
  color: #e2e8f0;
  border-top: 1px solid rgba(56, 189, 248, 0.3);
  z-index: 6;
}

.annotated-bar-info {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex-wrap: wrap;
}

.annotated-sub-note {
  color: #34d399;
  font-weight: 700;
}

.annotated-bar-actions {
  display: flex;
  align-items: center;
  gap: 0.45rem;
}

.btn-xs {
  padding: 0.25rem 0.6rem;
  font-size: 0.72rem;
  font-weight: 700;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-success {
  background: #10b981;
  color: #ffffff;
  border: 1px solid #059669;
}
.btn-success:hover {
  background: #059669;
}

.btn-outline-secondary {
  background: rgba(30, 41, 59, 0.8);
  color: #94a3b8;
  border: 1px solid #475569;
}
.btn-outline-secondary:hover {
  background: rgba(51, 65, 85, 0.9);
  color: #f1f5f9;
}

.btn-sm {
  padding: 0.35rem 0.7rem;
  font-size: 0.78rem;
}

.recognize-photo-upload-wrap {
  width: 100%;
}

.btn-accent-group {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  color: #fff;
  padding: 0.75rem 1rem;
  font-size: 0.88rem;
  font-weight: 700;
  border-radius: 6px;
  cursor: pointer;
  text-align: center;
  transition: all 0.2s ease;
}
.btn-accent-group:hover {
  background: linear-gradient(135deg, #4f46e5, #7c3aed);
}
.btn-accent-group.disabled {
  opacity: 0.6;
  pointer-events: none;
}

/* Target Summary Banner (Michael Pinpointing) */
.target-pinpoint-banner {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 1.25rem;
  border-radius: 10px;
  margin-top: 1rem;
  margin-bottom: 1.25rem;
  border: 2px solid transparent;
  transition: all 0.3s ease;
}
.target-pinpoint-banner.is-detected {
  background: rgba(16, 185, 129, 0.12);
  border-color: #10b981;
  box-shadow: 0 0 25px rgba(16, 185, 129, 0.25);
}
.target-pinpoint-banner.is-not-detected {
  background: rgba(100, 116, 139, 0.12);
  border-color: #64748b;
}
.target-pinpoint-icon {
  font-size: 2rem;
  line-height: 1;
}
.target-pinpoint-text {
  flex: 1;
}
.target-pinpoint-heading {
  margin: 0 0 0.25rem 0;
  font-size: 1.05rem;
  font-weight: 700;
  color: #fff;
}
.target-pinpoint-msg {
  margin: 0;
  font-size: 0.92rem;
  color: #e2e8f0;
}
.target-pinpoint-badge {
  padding: 0.45rem 0.9rem;
  background: #10b981;
  color: #ffffff;
  font-size: 0.85rem;
  font-weight: 800;
  border-radius: 20px;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
  white-space: nowrap;
}

.face-comparison-card.face-is-target {
  border-color: #10b981;
  box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
  background: rgba(16, 185, 129, 0.06);
}

.face-comparison-card.face-is-unmatched {
  border-color: rgba(239, 68, 68, 0.65);
  background: rgba(239, 68, 68, 0.05);
}

.face-target-chip {
  position: absolute;
  top: -8px;
  left: 50%;
  transform: translateX(-50%);
  background: #10b981;
  color: #ffffff;
  font-size: 0.62rem;
  font-weight: 800;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  box-shadow: 0 2px 6px rgba(16, 185, 129, 0.5);
  white-space: nowrap;
}

.face-unmatched-chip {
  position: absolute;
  top: -8px;
  left: 50%;
  transform: translateX(-50%);
  background: #ef4444;
  color: #ffffff;
  font-size: 0.6rem;
  font-weight: 800;
  padding: 0.15rem 0.42rem;
  border-radius: 4px;
  box-shadow: 0 2px 6px rgba(239, 68, 68, 0.45);
  white-space: nowrap;
}

.michael-winner-banner {
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
  border: 1px solid #10b981;
  font-size: 0.76rem;
  font-weight: 700;
  padding: 0.25rem 0.5rem;
  border-radius: 5px;
  margin-top: 0.25rem;
  margin-bottom: 0.25rem;
}

.michael-unmatched-banner {
  background: rgba(239, 68, 68, 0.18);
  color: #fca5a5;
  border: 1px solid #ef4444;
  font-size: 0.75rem;
  font-weight: 700;
  padding: 0.25rem 0.5rem;
  border-radius: 5px;
  margin-top: 0.25rem;
  margin-bottom: 0.25rem;
}

.michael-other-banner {
  background: rgba(100, 116, 139, 0.15);
  color: #94a3b8;
  font-size: 0.72rem;
  font-style: italic;
  padding: 0.2rem 0.45rem;
  border-radius: 4px;
  margin-top: 0.25rem;
  margin-bottom: 0.25rem;
}

/* Kartu Pengkotakan Per Wajah & Perbandingan Personil */
.result-header-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.75rem;
}

.detected-faces-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 1rem;
}

.face-comparison-card {
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid rgba(56, 189, 248, 0.3);
  border-radius: 10px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.85rem;
}

.face-card-top {
  display: flex;
  align-items: center;
  gap: 0.9rem;
}

.face-thumb-box {
  position: relative;
  flex-shrink: 0;
}

.face-thumb-img {
  width: 74px;
  height: 74px;
  border-radius: 8px;
  object-fit: cover;
  border: 3px solid #10b981;
  display: block;
}

.face-index-tag {
  position: absolute;
  bottom: -6px;
  left: 50%;
  transform: translateX(-50%);
  font-size: 0.65rem;
  font-weight: 800;
  color: #fff;
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
  white-space: nowrap;
}

.face-match-summary {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.compared-label {
  font-size: 0.72rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.personil-winner-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.personil-winner-name {
  font-size: 1.15rem;
  color: #fff;
}

.personil-winner-pct {
  font-size: 0.78rem;
  font-weight: 800;
  color: #fff;
  padding: 0.15rem 0.5rem;
  border-radius: 5px;
}

.personil-role-sub {
  font-size: 0.75rem;
  color: #94a3b8;
}

.runner-up-note {
  font-size: 0.74rem;
  color: #cbd5e1;
  background: rgba(255, 255, 255, 0.05);
  padding: 0.2rem 0.45rem;
  border-radius: 4px;
  margin-top: 0.15rem;
}

.personil-bars-section {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  border-top: 1px solid var(--border);
  padding-top: 0.65rem;
}

.bars-title {
  font-size: 0.75rem;
  font-weight: 700;
  color: #cbd5e1;
}

.personil-bars-list {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}

.personil-bar-item {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding: 0.35rem 0.5rem;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.02);
}

.personil-bar-item.is-top {
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.25);
}

.personil-bar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
}

.personil-cand-name {
  display: flex;
  align-items: center;
  gap: 0.35rem;
}

.rank-num {
  font-size: 0.7rem;
  font-weight: 700;
  color: #38bdf8;
}

.personil-cand-Right {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.dist-tag {
  color: var(--text-muted);
  font-size: 0.7rem;
}

.frame-mode-subtitle {
  font-size: 0.78rem;
  color: var(--text-muted);
  display: block;
  margin-top: 0.15rem;
}

.frame-mode-switcher {
  display: flex;
  gap: 0.5rem;
  background: var(--bg-main);
  padding: 0.3rem;
  border-radius: 8px;
  border: 1px solid var(--border);
}

.frame-switch-btn {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 0.5rem 0.75rem;
  font-size: 0.82rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.frame-switch-btn.active {
  background: #0284c7;
  color: #fff;
  box-shadow: 0 2px 8px rgba(2, 132, 199, 0.35);
}

.focus-mode-switcher {
  display: flex;
  gap: 0.5rem;
  background: var(--bg-main);
  padding: 0.3rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  margin-top: -0.35rem;
}

.focus-switch-btn {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 0.45rem 0.65rem;
  font-size: 0.8rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.focus-switch-btn.active {
  background: #10b981;
  color: #fff;
  box-shadow: 0 2px 8px rgba(16, 185, 129, 0.35);
}

.face-target-guide {
  position: absolute;
  inset: 0;
  pointer-events: none;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 3;
}

/* BINGKAI KARTU SEDERHANA (Transparan & Bersih) */
.ktp-card-frame {
  width: 80%;
  aspect-ratio: 1.586 / 1;
  position: relative;
  border: 1.5px solid rgba(56, 189, 248, 0.5);
  border-radius: 16px;
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 10px;
}

.ktp-photo-box-guide {
  position: absolute;
  left: 62%;
  top: 18%;
  width: 32%;
  height: 66%;
  border: 1.5px dashed rgba(16, 185, 129, 0.8);
  border-radius: 8px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 6px;
}

.ktp-photo-label {
  font-size: 0.68rem;
  color: #34d399;
  background: rgba(15, 23, 42, 0.75);
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
}

/* Preview Crop Bar di bawah kamera */
.crop-preview-bar {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  background: rgba(15, 23, 42, 0.75);
  border: 1px solid rgba(56, 189, 248, 0.3);
  padding: 0.65rem 0.85rem;
  border-radius: 8px;
}

.crop-preview-img {
  width: 56px;
  height: 56px;
  border-radius: 6px;
  object-fit: cover;
  border: 2px solid #10b981;
  flex-shrink: 0;
}

.crop-preview-info {
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
}

.crop-preview-title {
  font-size: 0.82rem;
  font-weight: 700;
  color: #e2e8f0;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}

.method-pill {
  font-size: 0.68rem;
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
  padding: 0.1rem 0.45rem;
  border-radius: 4px;
}

.crop-preview-desc {
  font-size: 0.73rem;
  color: var(--text-muted);
}

.btn-accent-ktp {
  background: linear-gradient(135deg, #0284c7, #2563eb);
  color: #fff;
  padding: 0.8rem 1rem;
  font-size: 0.9rem;
  font-weight: 700;
  border-radius: 8px;
  box-shadow: 0 3px 10px rgba(37, 99, 235, 0.3);
}
.btn-accent-ktp:hover {
  background: linear-gradient(135deg, #0369a1, #1d4ed8);
}

.btn-outline-info {
  background: rgba(2, 132, 199, 0.12);
  border: 1px solid #38bdf8;
  color: #38bdf8;
  padding: 0.65rem 1rem;
}
.btn-outline-info:hover {
  background: rgba(2, 132, 199, 0.22);
}

.or-divider {
  display: flex;
  align-items: center;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.or-divider::before,
.or-divider::after {
  content: '';
  flex: 1;
  border-bottom: 1px solid var(--border);
}
.or-divider span {
  padding: 0 0.6rem;
}

.ktp-cropped-preview-box {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  background: rgba(16, 185, 129, 0.1);
  border: 1px solid rgba(16, 185, 129, 0.35);
  padding: 0.65rem;
  border-radius: 8px;
}

.ktp-cropped-img {
  width: 60px;
  height: 60px;
  object-fit: cover;
  border-radius: 6px;
  border: 2px solid #10b981;
  flex-shrink: 0;
}

.ktp-cropped-meta {
  display: flex;
  flex-direction: column;
  gap: 0.15rem;
  font-size: 0.78rem;
}
.ktp-cropped-meta strong {
  color: #34d399;
}
.ktp-cropped-meta small {
  color: var(--text-muted);
  font-size: 0.72rem;
}

.reticle-box {
  width: 220px;
  height: 270px;
  position: relative;
  border-radius: 50px;
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: flex-end;
  justify-content: center;
  padding-bottom: 12px;
}

.corner {
  position: absolute;
  width: 28px;
  height: 28px;
  border-color: #38bdf8;
  border-style: solid;
}

.corner.tl {
  top: 0;
  left: 0;
  border-width: 3px 0 0 3px;
  border-top-left-radius: 20px;
}
.corner.tr {
  top: 0;
  right: 0;
  border-width: 3px 3px 0 0;
  border-top-right-radius: 20px;
}
.corner.bl {
  bottom: 0;
  left: 0;
  border-width: 0 0 3px 3px;
  border-bottom-left-radius: 20px;
}
.corner.br {
  bottom: 0;
  right: 0;
  border-width: 0 3px 3px 0;
  border-bottom-right-radius: 20px;
}

.target-text {
  font-size: 0.75rem;
  color: #38bdf8;
  background: rgba(15, 23, 42, 0.8);
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
}

.camera-off-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  color: var(--text-muted);
}

.camera-off-icon {
  font-size: 3rem;
}

.threshold-control-row {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  font-size: 0.85rem;
}

.camera-actions-col {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.action-buttons-stack {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.security-note {
  font-size: 0.8rem;
  color: #34d399;
  background: rgba(16, 185, 129, 0.1);
  padding: 0.6rem 0.8rem;
  border-radius: 6px;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

/* Source selector pills */
.source-pills-row {
  display: flex;
  gap: 0.5rem;
  background: var(--bg-main);
  padding: 0.25rem;
  border-radius: 8px;
  border: 1px solid var(--border);
}

.source-pill {
  flex: 1;
  background: transparent;
  border: none;
  color: var(--text-muted);
  padding: 0.5rem 0.75rem;
  font-size: 0.82rem;
  font-weight: 600;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.source-pill.active {
  background: #3b82f6;
  color: #fff;
}

/* KTP Dropzone */
.ktp-upload-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.file-input-hidden {
  display: none;
}

.file-dropzone-label {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1.5rem 1rem;
  border: 2px dashed #3b82f6;
  background: rgba(59, 130, 246, 0.05);
  border-radius: 8px;
  cursor: pointer;
  text-align: center;
  gap: 0.35rem;
  transition: all 0.2s ease;
}

.file-dropzone-label:hover {
  background: rgba(59, 130, 246, 0.12);
  border-color: #60a5fa;
}

.dropzone-icon {
  font-size: 2rem;
}

.dropzone-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: #fff;
}

.dropzone-sub {
  font-size: 0.75rem;
  color: var(--text-muted);
}

/* Result Card */
.result-card {
  border-left: 4px solid #3b82f6;
}

.verdict-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.matched-name-title {
  font-size: 1.5rem;
  font-weight: 800;
  color: #fff;
  margin-top: 0.25rem;
}

.matched-role {
  font-size: 0.95rem;
  color: #a7f3d0;
  font-weight: 500;
}

.badge-time {
  font-size: 0.75rem;
  background: rgba(0, 0, 0, 0.3);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}

/* General Layout Elements */
.tab-content {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.grid-2col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

@media (max-width: 900px) {
  .grid-2col {
    grid-template-columns: 1fr;
  }
}

.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.card-header-flex {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.card-title {
  font-size: 1.15rem;
  font-weight: 700;
}

.card-desc {
  font-size: 0.88rem;
  color: var(--text-muted);
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.form-item {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.form-item label {
  font-size: 0.88rem;
  font-weight: 600;
}

.form-control {
  background: var(--bg-main);
  border: 1px solid var(--border);
  color: #fff;
  padding: 0.65rem 0.85rem;
  border-radius: 6px;
  font-size: 0.9rem;
}

.form-control:focus {
  outline: none;
  border-color: #3b82f6;
}

.slider-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.threshold-badge {
  background: #3b82f6;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.82rem;
  font-weight: 700;
}

.slider {
  width: 100%;
  accent-color: #3b82f6;
}

/* Buttons */
.btn {
  padding: 0.65rem 1.1rem;
  font-size: 0.88rem;
  font-weight: 600;
  border-radius: 6px;
  border: none;
  transition: all 0.2s ease;
}

.btn-lg {
  padding: 0.85rem 1.25rem;
  font-size: 1rem;
}

.btn-primary {
  background: #3b82f6;
  color: #fff;
}
.btn-primary:hover {
  background: #2563eb;
}

.btn-secondary {
  background: var(--bg-card-subtle);
  color: #fff;
}
.btn-secondary:hover {
  background: #475569;
}

.btn-success {
  background: #10b981;
  color: #fff;
}
.btn-success:hover {
  background: #059669;
}

.btn-danger {
  background: #ef4444;
  color: #fff;
}
.btn-danger:hover {
  background: #dc2626;
}

.btn-outline-danger {
  background: transparent;
  border: 1px solid #ef4444;
  color: #ef4444;
  padding: 0.4rem 0.8rem;
  font-size: 0.8rem;
}
.btn-outline-danger:hover {
  background: rgba(239, 68, 68, 0.15);
}

.btn-block {
  width: 100%;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* Verdict badge */
.verdict-badge {
  padding: 1.2rem;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.verdict-badge.match {
  background: rgba(16, 185, 129, 0.15);
  border: 1px solid #10b981;
}

.verdict-badge.no-match {
  background: rgba(239, 68, 68, 0.15);
  border: 1px solid #ef4444;
}

.verdict-icon {
  font-size: 1.1rem;
  font-weight: 700;
}

.verdict-badge.match .verdict-icon {
  color: #10b981;
}

.verdict-badge.no-match .verdict-icon {
  color: #ef4444;
}

.verdict-message {
  font-size: 0.9rem;
}

/* Metrics grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.75rem;
}

.metric-card {
  background: var(--bg-main);
  padding: 0.75rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  text-align: center;
}

.metric-title {
  font-size: 0.75rem;
  color: var(--text-muted);
}

.metric-val {
  font-size: 1.25rem;
  font-weight: 700;
}

.metric-val.highlight {
  color: #3b82f6;
}

.metric-note {
  font-size: 0.7rem;
  color: var(--text-muted);
}

/* Circular Gauge Card */
.percentage-gauge-card {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  background: linear-gradient(135deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95));
  border: 1px solid var(--border);
  padding: 1.25rem 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
}

@media (max-width: 650px) {
  .percentage-gauge-card {
    flex-direction: column;
    text-align: center;
  }
}

.gauge-circle-wrap {
  position: relative;
  width: 110px;
  height: 110px;
  flex-shrink: 0;
}

.gauge-svg {
  width: 100%;
  height: 100%;
  transform: rotate(-90deg);
}

.gauge-track {
  fill: none;
  stroke: rgba(255, 255, 255, 0.08);
  stroke-width: 10;
}

.gauge-progress {
  fill: none;
  stroke-width: 10;
  stroke-linecap: round;
  transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.4s ease;
}

.gauge-center-content {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.gauge-big-num {
  font-size: 1.6rem;
  font-weight: 800;
  line-height: 1.1;
  font-family: system-ui, sans-serif;
}

.gauge-mini-label {
  font-size: 0.68rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.gauge-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
}

.gauge-status-title {
  font-size: 1.25rem;
  font-weight: 800;
}

.gauge-detail-desc {
  font-size: 0.88rem;
  color: #cbd5e1;
}

.gauge-legend {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin-top: 0.25rem;
}

.legend-badge {
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
}

.legend-badge.red {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
}
.legend-badge.yellow {
  background: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
}
.legend-badge.green {
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
}

/* Tables */
.candidates-table-wrap {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.candidates-table-wrap h4 {
  font-size: 0.9rem;
  font-weight: 600;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.data-table th, .data-table td {
  padding: 0.6rem 0.75rem;
  text-align: left;
  border-bottom: 1px solid var(--border);
}

.data-table th {
  background: var(--bg-main);
  color: var(--text-muted);
  font-weight: 600;
}

.row-winner {
  background: rgba(16, 185, 129, 0.1);
}

.vector-preview-box {
  background: var(--bg-main);
  padding: 0.75rem;
  border-radius: 6px;
  font-size: 0.75rem;
  border: 1px solid var(--border);
}

.vector-title {
  display: block;
  color: var(--text-muted);
  margin-bottom: 0.3rem;
}

.vector-preview-box code {
  color: #38bdf8;
  word-break: break-all;
}

/* Badges */
.badge {
  display: inline-block;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
}

.badge-info {
  background: #0284c7;
  color: #fff;
}

.badge-success {
  background: rgba(16, 185, 129, 0.2);
  color: #34d399;
}

.badge-danger {
  background: rgba(239, 68, 68, 0.2);
  color: #f87171;
}

.badge-warning {
  background: rgba(245, 158, 11, 0.2);
  color: #fbbf24;
}

/* Percentage progress bars & candidate highlights */
.similarity-progress-wrap {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  min-width: 140px;
}

.progress-bar-bg {
  flex: 1;
  height: 8px;
  background: var(--bg-main);
  border-radius: 9999px;
  overflow: hidden;
  border: 1px solid var(--border);
}

.progress-bar-fill {
  height: 100%;
  border-radius: 9999px;
  transition: width 0.3s ease;
}

.percentage-text {
  font-weight: 700;
  font-size: 0.88rem;
  min-width: 48px;
}

.top-candidate-banner {
  margin-top: 0.5rem;
  padding: 0.75rem 1rem;
  background: rgba(15, 23, 42, 0.6);
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.top-candidate-box {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.top-candidate-text {
  font-size: 0.95rem;
  color: #e2e8f0;
}

.top-percentage-badge {
  font-size: 1.1rem;
  font-weight: 800;
  padding: 0.25rem 0.75rem;
  border-radius: 6px;
}

.top-percentage-badge.match {
  background: #10b981;
  color: #fff;
}

.top-percentage-badge.no-match {
  background: #ef4444;
  color: #fff;
}

/* Profile Items */
.profiles-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  max-height: 550px;
  overflow-y: auto;
}

.profile-item {
  display: flex;
  align-items: center;
  gap: 1rem;
  background: var(--bg-main);
  padding: 0.85rem;
  border-radius: 8px;
  border: 1px solid var(--border);
}

.avatar-circle {
  width: 52px;
  height: 52px;
  border-radius: 50%;
  background: #3b82f6;
  color: #fff;
  font-size: 1.3rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.avatar-circle.sm {
  width: 40px;
  height: 40px;
  font-size: 1rem;
}

.profile-details {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.profile-title-row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.profile-title-row h3 {
  font-size: 0.95rem;
}

.profile-meta {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.storage-tag {
  font-size: 0.75rem;
  color: #34d399;
}

.vector-snippet {
  font-size: 0.7rem;
  color: #94a3b8;
  font-family: monospace;
}

.btn-icon {
  background: transparent;
  border: none;
  font-size: 1rem;
  padding: 0.4rem;
  border-radius: 4px;
}
.btn-icon:hover {
  background: rgba(239, 68, 68, 0.2);
}

/* Storage Stats */
.stats-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
}

@media (max-width: 768px) {
  .stats-row {
    grid-template-columns: 1fr;
  }
}

.stat-card {
  padding: 1.25rem;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
}

.stat-card.danger {
  background: rgba(239, 68, 68, 0.12);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

.stat-card.success {
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.3);
}

.stat-card.highlight {
  background: rgba(59, 130, 246, 0.12);
  border: 1px solid rgba(59, 130, 246, 0.3);
}

.stat-label {
  font-size: 0.82rem;
  color: var(--text-muted);
}

.stat-value {
  font-size: 1.8rem;
  font-weight: 800;
}

.stat-card.danger .stat-value {
  color: #f87171;
}

.stat-card.success .stat-value {
  color: #34d399;
}

.stat-card.highlight .stat-value {
  color: #60a5fa;
}

.stat-sub {
  font-size: 0.78rem;
  color: var(--text-muted);
}

.callout {
  padding: 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
}

.callout-warning {
  background: rgba(245, 158, 11, 0.1);
  border-left: 3px solid #f59e0b;
  color: #fbbf24;
}

.alert {
  padding: 0.75rem 1rem;
  border-radius: 6px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.88rem;
}

.alert-success {
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
  border: 1px solid #10b981;
}

.alert-danger {
  background: rgba(239, 68, 68, 0.15);
  color: #f87171;
  border: 1px solid #ef4444;
}

.close-btn {
  background: transparent;
  border: none;
  color: inherit;
  font-size: 1rem;
}

.empty-state {
  text-align: center;
  padding: 2rem;
  color: var(--text-muted);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
}

.placeholder-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  padding: 3rem 1.5rem;
  border: 2px dashed var(--border);
  border-radius: 8px;
  color: var(--text-muted);
  gap: 0.75rem;
}

.placeholder-icon {
  font-size: 2.5rem;
}
</style>
