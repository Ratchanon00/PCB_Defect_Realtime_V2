

let currentAppMode = 'camera'; 
let currentDetectionResult = null;
let currentImageResult = null;
let currentSource = null;
let currentViewMode = 'annotated';

let zoomScale = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let startPanX = 0;
let startPanY = 0;

let cameraStream = null;
let isCameraActive = false;
let currentCameraType = 'directshow';
let telemetryIntervalId = null;

let isCalibrationMode = false;
let calibrationCorners = [];
const CALIB_LABELS = ['บนซ้าย (TL)', 'บนขวา (TR)', 'ล่างขวา (BR)', 'ล่างซ้าย (BL)'];
const CALIB_COLORS = ['#ef4444', '#f59e0b', '#10b981', '#06b6d4'];

const modeLiveCamBtn = document.getElementById('modeLiveCamBtn');
const modeImageUploadBtn = document.getElementById('modeImageUploadBtn');
const cameraControlCard = document.getElementById('cameraControlCard');
const imageUploadCard = document.getElementById('imageUploadCard');

const cameraStatusBadge = document.getElementById('cameraStatusBadge');
const startCameraBtn = document.getElementById('startCameraBtn');
const stopCameraBtn = document.getElementById('stopCameraBtn');
const liveDetectToggle = document.getElementById('liveDetectToggle');
const quickStartCamBtn = document.getElementById('quickStartCamBtn');
const cameraStreamContainer = document.getElementById('cameraStreamContainer');
const cameraStreamImg = document.getElementById('cameraStreamImg');
const cameraVideo = document.getElementById('cameraVideo');
const cameraHudBadge = document.getElementById('cameraHudBadge');
const hudStatusText = document.getElementById('hudStatusText');
const hudResolutionText = document.getElementById('hudResolutionText');
const hudFpsText = document.getElementById('hudFpsText');

const calibrationPanel = document.getElementById('calibrationPanel');
const calibStatusText = document.getElementById('calibStatusText');
const calibPreviewRow = document.getElementById('calibPreviewRow');
const calibRefThumb = document.getElementById('calibRefThumb');
const calibInfoLabel = document.getElementById('calibInfoLabel');
const calibInfoDate = document.getElementById('calibInfoDate');
const startCalibBtn = document.getElementById('startCalibBtn');
const calibActionsRow = document.getElementById('calibActionsRow');
const saveCalibBtn = document.getElementById('saveCalibBtn');
const resetCalibBtn = document.getElementById('resetCalibBtn');
const cancelCalibBtn = document.getElementById('cancelCalibBtn');
const clearCalibBtn = document.getElementById('clearCalibBtn');
const calibrationCanvas = document.getElementById('calibrationCanvas');
const calibCtx = calibrationCanvas ? calibrationCanvas.getContext('2d') : null;
const calibBanner = document.getElementById('calibBanner');
const calibBannerText = document.getElementById('calibBannerText');
const miniCancelCalibBtn = document.getElementById('miniCancelCalibBtn');

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const modelSelect = document.getElementById('modelSelect');
const confRange = document.getElementById('confRange');
const confVal = document.getElementById('confVal');
const iouRange = document.getElementById('iouRange');
const iouVal = document.getElementById('iouVal');
const reRunBtn = document.getElementById('reRunBtn');
const samplesCarousel = document.getElementById('samplesCarousel');
const sampleBarSection = document.getElementById('sampleBarSection');
const toggleSamplesBtn = document.getElementById('toggleSamplesBtn');
const systemDeviceBadge = document.getElementById('systemDeviceBadge');

const viewportStage = document.getElementById('viewportStage');
const stageContainer = document.getElementById('stageContainer');
const mainDisplayImg = document.getElementById('mainDisplayImg');
const splitViewContainer = document.getElementById('splitViewContainer');
const splitOriginalImg = document.getElementById('splitOriginalImg');
const splitAnnotatedImg = document.getElementById('splitAnnotatedImg');
const scannerLaser = document.getElementById('scannerLaser');
const processingOverlay = document.getElementById('processingOverlay');
const emptyState = document.getElementById('emptyState');

const verdictBanner = document.getElementById('verdictBanner');
const verdictIcon = document.getElementById('verdictIcon');
const verdictTitle = document.getElementById('verdictTitle');
const verdictSub = document.getElementById('verdictSub');
const metricDefectCount = document.getElementById('metricDefectCount');
const metricInferenceTime = document.getElementById('metricInferenceTime');
const metricFps = document.getElementById('metricFps');
const cropsGallery = document.getElementById('cropsGallery');
const defectTableBody = document.getElementById('defectTableBody');
const defectCountBadge = document.getElementById('defectCountBadge');

const imageNameBadge = document.getElementById('imageNameBadge');
const imageDimBadge = document.getElementById('imageDimBadge');

const offscreenCanvas = document.createElement('canvas');
const offscreenCtx = offscreenCanvas.getContext('2d', { willReadFrequently: true });

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSystem();
  setupEventListeners();
});

async function initSystem() {
  try {
    if (confRange && confVal) {
      confRange.value = 50;
      confVal.textContent = '50%';
    }

    const statusRes = await fetch('/api/status');
    const statusData = await statusRes.json();
    if (systemDeviceBadge && statusData.device) {
      systemDeviceBadge.textContent = statusData.cuda_available ? 'CUDA GPU' : 'CPU MODE';
    }

    const modelsRes = await fetch('/api/models');
    const modelsData = await modelsRes.json();
    modelSelect.innerHTML = '';
    modelsData.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.path;
      opt.textContent = `${m.name} (${m.size_mb} MB)`;
      if (m.path === modelsData.active_model) {
        opt.selected = true;
      }
      modelSelect.appendChild(opt);
    });

    const statsRes = await fetch('/api/dataset-stats');
    const statsData = await statsRes.json();
    document.getElementById('statTrainCount').textContent = statsData.train_count;
    document.getElementById('statValCount').textContent = statsData.valid_count;

    await loadSamples();
    await loadCalibrationStatus();

    
    setAppMode('camera');

  } catch (err) {
    console.error('Initialization error:', err);
  }
}

function setAppMode(mode) {
  currentAppMode = mode;

  const emptyTitle = document.getElementById('emptyStateTitle');

  if (mode === 'camera') {
    modeLiveCamBtn.classList.add('active');
    modeImageUploadBtn.classList.remove('active');
    cameraControlCard.style.display = 'block';
    imageUploadCard.style.display = 'none';
    stageContainer.style.display = 'none';

    const viewModeContainer = document.getElementById('viewModeGroupContainer');
    const reRunBtn = document.getElementById('reRunBtn');
    if (viewModeContainer) viewModeContainer.style.display = 'none';
    if (reRunBtn) reRunBtn.style.display = 'none';

    // ซ่อนภาพซูมจุดบกพร่องในโหมดกล้องสด (ใช้ไม่ได้ในโหมดนี้)
    const defectCropsCard = document.getElementById('defectCropsCard');
    if (defectCropsCard) defectCropsCard.style.display = 'none';

    if (sampleBarSection) sampleBarSection.classList.add('collapsed');

    if (emptyTitle) emptyTitle.textContent = 'กล้องยังไม่ได้เปิดใช้งาน';

    if (isCameraActive) {
      cameraStreamContainer.style.display = 'flex';
      emptyState.style.display = 'none';
      imageNameBadge.textContent = 'กล้องสด (Live Camera Feed)';
      imageDimBadge.textContent = '1280 × 720 px';
    } else {
      cameraStreamContainer.style.display = 'none';
      emptyState.style.display = 'flex';
      if (quickStartCamBtn) quickStartCamBtn.style.display = 'inline-flex';
      imageNameBadge.textContent = 'กล้องสด (กล้องปิดอยู่)';
      imageDimBadge.textContent = '0 × 0 px';
    }

  } else {
    
    modeImageUploadBtn.classList.add('active');
    modeLiveCamBtn.classList.remove('active');
    imageUploadCard.style.display = 'block';
    cameraControlCard.style.display = 'none';
    cameraStreamContainer.style.display = 'none';

    
    if (isCameraActive) {
      stopCamera();
    }

    const viewModeContainer = document.getElementById('viewModeGroupContainer');
    const reRunBtn = document.getElementById('reRunBtn');
    if (viewModeContainer) viewModeContainer.style.display = 'block';
    if (reRunBtn) reRunBtn.style.display = 'flex';

    if (sampleBarSection) sampleBarSection.classList.remove('collapsed');

    // แสดงภาพซูมจุดบกพร่องกลับมาในโหมดรูปภาพ
    const defectCropsCard = document.getElementById('defectCropsCard');
    if (defectCropsCard) defectCropsCard.style.display = '';

    
    if (currentImageResult && (currentImageResult.annotated_image || currentImageResult.original_image)) {
      stageContainer.style.display = 'flex';
      emptyState.style.display = 'none';
      displayResults(currentImageResult);
    } else {
      stageContainer.style.display = 'none';
      mainDisplayImg.removeAttribute('src');
      mainDisplayImg.style.display = 'none';
      splitViewContainer.style.display = 'none';
      emptyState.style.display = 'flex';
      if (quickStartCamBtn) quickStartCamBtn.style.display = 'none';
      if (emptyTitle) emptyTitle.textContent = 'ยังไม่ได้เลือกภาพ PCB';
      imageNameBadge.textContent = 'โหมดรูปภาพ (รอเลือกภาพ)';
      imageDimBadge.textContent = '0 × 0 px';

      
      verdictBanner.className = 'verdict-banner idle';
      verdictIcon.textContent = '⏳';
      verdictTitle.textContent = 'รอเริ่มการตรวจจับ';
      metricDefectCount.textContent = '0';
      metricInferenceTime.textContent = '--';
      metricFps.textContent = '--';
      defectCountBadge.textContent = '0 จุด';
      cropsGallery.innerHTML = '<p class="empty-crops-msg">ยังไม่มีจุดบกพร่องที่ตรวจพบ</p>';
      defectTableBody.innerHTML = '<tr><td colspan="5" class="empty-td">ไม่มีข้อมูลการตรวจพบ</td></tr>';
    }
  }
}

async function startCamera() {
  if (isCameraActive) return;
  await startDirectShowCamera();
}

async function startDirectShowCamera() {
  processingOverlay.style.display = 'flex';
  document.querySelector('.processing-text').textContent = 'กำลังเปิดกล้อง USB (DirectShow)...';

  try {
    const formData = new FormData();
    formData.append('camera_index', '0');
    formData.append('conf_thresh', (parseFloat(confRange.value) / 100).toString());
    formData.append('iou_thresh', (parseFloat(iouRange.value) / 100).toString());
    if (modelSelect.value) {
      formData.append('model_path', modelSelect.value);
    }

    const startRes = await fetch('/api/camera/start', {
      method: 'POST',
      body: formData
    });

    if (!startRes.ok) {
      throw new Error('ไม่สามารถเริ่มสัญญาณกล้อง USB ได้');
    }

    cameraStreamContainer.style.display = 'flex';
    stageContainer.style.display = 'none';
    emptyState.style.display = 'none';

    
    cameraStreamImg.style.display = 'block';
    cameraVideo.style.display = 'none';
    cameraStreamImg.onerror = () => {
      console.warn('Camera stream connection retry...');
      if (isCameraActive) {
        setTimeout(() => {
          if (isCameraActive) {
            cameraStreamImg.src = '/api/camera/stream?t=' + Date.now();
          }
        }, 1000);
      }
    };
    cameraStreamImg.src = '/api/camera/stream?t=' + Date.now();

    isCameraActive = true;
    currentCameraType = 'directshow';
    updateCameraUiState(true);

    hudStatusText.textContent = 'USB CAM (DIRECT)';
    hudResolutionText.textContent = '1280×720';

    
    startTelemetryPolling();

  } catch (err) {
    console.error('DirectShow Camera error:', err);
    alert('ไม่สามารถเปิดกล้องได้: ' + err.message + '\n\nคำแนะนำ:\n1. ตรวจสอบว่าสาย USB กล้องเสียบแน่นแล้ว\n2. ตรวจสอบว่าไม่มีโปรแกรมอื่น (เช่น Camera App, OBS) กำลังใช้งานกล้องอยู่');
    updateCameraUiState(false);
  } finally {
    processingOverlay.style.display = 'none';
  }
}

function startTelemetryPolling() {
  if (telemetryIntervalId) clearInterval(telemetryIntervalId);
  telemetryIntervalId = setInterval(async () => {
    if (!isCameraActive || currentCameraType !== 'directshow') {
      clearInterval(telemetryIntervalId);
      return;
    }
    try {
      const res = await fetch('/api/camera/telemetry');
      if (res.ok) {
        const data = await res.json();
        currentDetectionResult = data;
        updateTelemetryAndVerdict(data);
        if (data.fps) {
          hudFpsText.textContent = `${data.fps} FPS`;
        }
      }
    } catch (_) {}
  }, 250);
}

async function stopCamera() {
  if (currentCameraType === 'directshow') {
    try {
      await fetch('/api/camera/stop', { method: 'POST' });
    } catch (_) {}
    if (cameraStreamImg) cameraStreamImg.src = '';
    if (telemetryIntervalId) {
      clearInterval(telemetryIntervalId);
      telemetryIntervalId = null;
    }
  }

  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }

  isCameraActive = false;
  currentCameraType = null;
  cameraVideo.srcObject = null;

  updateCameraUiState(false);
}

function updateCameraUiState(active) {
  if (active) {
    startCameraBtn.style.display = 'none';
    stopCameraBtn.style.display = 'flex';
    cameraStatusBadge.className = 'badge badge-live active';
    cameraStatusBadge.textContent = '🟢 กำลังทำงาน';
    cameraStreamContainer.style.display = 'flex';
    stageContainer.style.display = 'none';
    cameraHudBadge.style.display = 'flex';
    emptyState.style.display = 'none';
    imageNameBadge.textContent = 'กล้องสด (Live Camera Feed)';
  } else {
    startCameraBtn.style.display = 'flex';
    stopCameraBtn.style.display = 'none';
    cameraStatusBadge.className = 'badge badge-live';
    cameraStatusBadge.textContent = '🔴 กล้องปิดอยู่';
    cameraStreamContainer.style.display = 'none';
    cameraHudBadge.style.display = 'none';
    scannerLaser.classList.remove('scanning');
    if (currentAppMode === 'camera') {
      emptyState.style.display = 'flex';
      stageContainer.style.display = 'none';
    }
    imageNameBadge.textContent = 'ยังไม่ได้เปิดกล้อง';
    hudFpsText.textContent = '-- FPS';
  }
}

async function syncCameraSettingsToBackend() {
  if (isCameraActive && currentCameraType === 'directshow') {
    const fd = new FormData();
    fd.append('conf_thresh', (parseFloat(confRange.value) / 100).toString());
    fd.append('iou_thresh', (parseFloat(iouRange.value) / 100).toString());
    if (modelSelect.value) fd.append('model_path', modelSelect.value);
    fd.append('auto_detect', liveDetectToggle.checked ? 'true' : 'false');
    const bVal = document.getElementById('cameraBrightnessRange')?.value || '0';
    fd.append('brightness', bVal);
    try {
      await fetch('/api/camera/settings', { method: 'POST', body: fd });
    } catch (_) {}
  }
}

function updateTelemetryAndVerdict(data) {
  verdictBanner.className = 'verdict-banner ' + (data.is_pass ? 'pass' : 'defect');
  if (data.is_pass) {
    verdictIcon.textContent = '✅';
    verdictTitle.textContent = 'PCB PASS (สมบูรณ์)';
    if (verdictSub) verdictSub.textContent = 'ไม่พบข้อบกพร่องตามระดับความไวที่กำหนด';
  } else {
    verdictIcon.textContent = '⚠️';
    verdictTitle.textContent = `DEFECT DETECTED (พบ ${data.defect_count} จุด)`;
    if (verdictSub) verdictSub.textContent = `พบข้อบกพร่องประเภท Missing Hole จำนวน ${data.defect_count} แห่ง`;
  }

  metricDefectCount.textContent = data.defect_count;
  metricInferenceTime.textContent = data.timing.inference_ms;
  metricFps.textContent = data.fps || data.timing.fps || '--';
  defectCountBadge.textContent = `${data.defect_count} จุด`;

  renderCrops(data.patches);
  renderTable(data.defects);
}

async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    samplesCarousel.innerHTML = '';

    if (!data.samples || data.samples.length === 0) {
      samplesCarousel.innerHTML = '<span style="color:var(--text-dim);font-size:0.8rem;">ไม่พบภาพตัวอย่าง</span>';
      return;
    }

    data.samples.forEach((sample) => {
      const card = document.createElement('div');
      card.className = 'sample-thumb-card';
      card.title = `คลิกเพื่อตรวจสอบ: ${sample.filename}`;
      card.innerHTML = `
        <img class="sample-thumb-img" src="/api/sample-image/${encodeURIComponent(sample.filename)}" alt="PCB" loading="lazy">
        <span class="sample-thumb-name">${sample.filename.replace('_jpg.rf.', '..')}</span>
      `;
      card.addEventListener('click', () => {
        document.querySelectorAll('.sample-thumb-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');

        setAppMode('image');
        currentSource = { type: 'sample', filename: sample.filename };
        runDetection();
      });
      samplesCarousel.appendChild(card);
    });

  } catch (err) {
    console.error('Error loading samples:', err);
  }
}

async function runDetection() {
  if (!currentSource) return;

  processingOverlay.style.display = 'flex';
  document.querySelector('.processing-text').textContent = 'AI กำลังสแกนหาข้อบกพร่องของ PCB...';
  scannerLaser.classList.add('scanning');
  emptyState.style.display = 'none';

  const formData = new FormData();
  formData.append('conf_thresh', (parseFloat(confRange.value) / 100).toString());
  formData.append('iou_thresh', (parseFloat(iouRange.value) / 100).toString());
  if (modelSelect.value) {
    formData.append('model_path', modelSelect.value);
  }

  if (currentSource.type === 'sample') {
    formData.append('sample_filename', currentSource.filename);
  } else if (currentSource.type === 'file') {
    formData.append('file', currentSource.file);
  }

  try {
    const res = await fetch('/api/detect', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'การตรวจสอบล้มเหลว');
    }

    const data = await res.json();
    currentDetectionResult = data;
    currentImageResult = data;
    displayResults(data);

  } catch (err) {
    alert('เกิดข้อผิดพลาดในการตรวจสอบ: ' + err.message);
    console.error(err);
  } finally {
    processingOverlay.style.display = 'none';
    scannerLaser.classList.remove('scanning');
  }
}

function displayResults(data) {
  imageNameBadge.textContent = data.filename || 'PCB Image';
  if (data.dimensions) {
    imageDimBadge.textContent = `${data.dimensions.width} × ${data.dimensions.height} px`;
  }

  updateTelemetryAndVerdict(data);
  renderCurrentView();
}

function renderCurrentView() {
  const data = currentImageResult || currentDetectionResult;
  if (!data || (!data.original_image && !data.annotated_image)) {
    stageContainer.style.display = 'none';
    if (currentAppMode === 'image') {
      emptyState.style.display = 'flex';
    }
    return;
  }

  emptyState.style.display = 'none';

  if (isCameraActive) {
    stageContainer.style.display = 'none';
    cameraStreamContainer.style.display = 'flex';
    return;
  }

  cameraStreamContainer.style.display = 'none';
  stageContainer.style.display = 'flex';

  if (currentViewMode === 'split') {
    mainDisplayImg.style.display = 'none';
    splitViewContainer.style.display = 'flex';
    if (data.original_image) splitOriginalImg.src = data.original_image;
    if (data.annotated_image) splitAnnotatedImg.src = data.annotated_image;
  } else {
    splitViewContainer.style.display = 'none';
    mainDisplayImg.style.display = 'block';
    const imgSrc = (currentViewMode === 'original') ? data.original_image : data.annotated_image;
    if (imgSrc) {
      mainDisplayImg.src = imgSrc;
    }
  }
}

function renderCrops(patches) {
  cropsGallery.innerHTML = '';

  if (!patches || patches.length === 0) {
    cropsGallery.innerHTML = '<p class="empty-crops-msg">ไม่พบจุดบกพร่องในภาพนี้</p>';
    return;
  }

  patches.forEach(p => {
    const card = document.createElement('div');
    card.className = 'crop-card';
    card.title = `จุดที่ #${p.id} (${p.confidence}%) พิกัด: [${p.box.join(', ')}]`;
    card.innerHTML = `
      <img class="crop-img" src="${p.image_b64}" alt="Defect #${p.id}">
      <div class="crop-info">
        <span class="crop-id">#${p.id}</span>
        <span class="crop-conf">${p.confidence}%</span>
      </div>
    `;

    card.addEventListener('mouseenter', () => highlightTableRow(p.id));
    card.addEventListener('mouseleave', () => unhighlightTableRow(p.id));
    cropsGallery.appendChild(card);
  });
}

function renderTable(defects) {
  defectTableBody.innerHTML = '';

  if (!defects || defects.length === 0) {
    defectTableBody.innerHTML = '<tr><td colspan="5" class="empty-td">ผ่านการตรวจสอบ (ไม่มีข้อบกพร่อง)</td></tr>';
    return;
  }

  defects.forEach(d => {
    const tr = document.createElement('tr');
    tr.id = `defect-row-${d.id}`;
    tr.innerHTML = `
      <td><strong>#${d.id}</strong></td>
      <td><span class="tag-red">${d.class_name}</span></td>
      <td>${(d.confidence * 100).toFixed(1)}%</td>
      <td>(${d.cx}, ${d.cy})</td>
      <td>${d.width}×${d.height}</td>
    `;
    defectTableBody.appendChild(tr);
  });
}

function highlightTableRow(id) {
  const row = document.getElementById(`defect-row-${id}`);
  if (row) {
    row.classList.add('highlighted');
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function unhighlightTableRow(id) {
  const row = document.getElementById(`defect-row-${id}`);
  if (row) {
    row.classList.remove('highlighted');
  }
}

function setupEventListeners() {
  
  modeLiveCamBtn.addEventListener('click', () => setAppMode('camera'));
  modeImageUploadBtn.addEventListener('click', () => setAppMode('image'));

  
  startCameraBtn.addEventListener('click', startCamera);
  if (quickStartCamBtn) quickStartCamBtn.addEventListener('click', startCamera);
  stopCameraBtn.addEventListener('click', stopCamera);

  
  if (startCalibBtn) startCalibBtn.addEventListener('click', enterCalibrationMode);
  if (saveCalibBtn) saveCalibBtn.addEventListener('click', saveCalibration);
  if (resetCalibBtn) resetCalibBtn.addEventListener('click', resetCalibrationCorners);
  if (cancelCalibBtn) cancelCalibBtn.addEventListener('click', exitCalibrationMode);
  if (miniCancelCalibBtn) miniCancelCalibBtn.addEventListener('click', exitCalibrationMode);
  if (clearCalibBtn) clearCalibBtn.addEventListener('click', clearCalibration);
  if (calibrationCanvas) calibrationCanvas.addEventListener('click', handleCalibrationCanvasClick);
  if (cameraStreamImg) {
    cameraStreamImg.addEventListener('click', (e) => {
      if (isCalibrationMode) handleCalibrationCanvasClick(e);
    });
  }

  liveDetectToggle.addEventListener('change', () => {
    syncCameraSettingsToBackend();
  });

  
  if (toggleSamplesBtn && sampleBarSection) {
    toggleSamplesBtn.addEventListener('click', () => {
      sampleBarSection.classList.toggle('collapsed');
    });
  }

  
  confRange.addEventListener('input', (e) => {
    confVal.textContent = `${e.target.value}%`;
    syncCameraSettingsToBackend();
  });
  iouRange.addEventListener('input', (e) => {
    iouVal.textContent = `${e.target.value}%`;
    syncCameraSettingsToBackend();
  });
  modelSelect.addEventListener('change', syncCameraSettingsToBackend);

  const cameraBrightnessRange = document.getElementById('cameraBrightnessRange');
  const cameraBrightnessVal = document.getElementById('cameraBrightnessVal');
  if (cameraBrightnessRange) {
    cameraBrightnessRange.addEventListener('input', (e) => {
      if (cameraBrightnessVal) cameraBrightnessVal.textContent = e.target.value;
      syncCameraSettingsToBackend();
    });
  }

  reRunBtn.addEventListener('click', () => {
    if (currentSource) {
      runDetection();
    }
  });

  
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  window.addEventListener('paste', (e) => {
    const items = (e.clipboardData || e.originalEvent.clipboardData).items;
    for (let item of items) {
      if (item.type.indexOf('image') !== -1) {
        const blob = item.getAsFile();
        handleSelectedFile(blob);
        break;
      }
    }
  });

  
  const modeBtns = document.querySelectorAll('#viewModeGroup .mode-btn');
  modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      modeBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentViewMode = btn.dataset.mode;
      renderCurrentView();
    });
  });

  
  const zoomInBtn = document.getElementById('zoomInBtn');
  const zoomOutBtn = document.getElementById('zoomOutBtn');
  const resetZoomBtn = document.getElementById('resetZoomBtn');

  if (zoomInBtn) {
    zoomInBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      updateZoom(0.25);
    });
  }
  if (zoomOutBtn) {
    zoomOutBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      updateZoom(-0.25);
    });
  }
  if (resetZoomBtn) {
    resetZoomBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetZoom();
    });
  }

  viewportStage.addEventListener('wheel', (e) => {
    if (currentAppMode === 'image') {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.2 : -0.2;
      updateZoom(delta);
    }
  }, { passive: false });

  viewportStage.addEventListener('mousedown', (e) => {
    if (currentAppMode === 'image' && zoomScale > 1) {
      e.preventDefault();
      isPanning = true;
      startPanX = e.clientX - panX;
      startPanY = e.clientY - panY;
      viewportStage.classList.add('grabbing');
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    panX = e.clientX - startPanX;
    panY = e.clientY - startPanY;
    applyTransform();
  });

  window.addEventListener('mouseup', () => {
    isPanning = false;
    viewportStage.classList.remove('grabbing');
  });

  document.getElementById('exportJsonBtn').addEventListener('click', exportJsonData);
}

function handleSelectedFile(file) {
  setAppMode('image');
  document.querySelectorAll('.sample-thumb-card').forEach(c => c.classList.remove('active'));
  currentSource = { type: 'file', file: file, name: file.name };
  runDetection();
}

function updateZoom(delta) {
  zoomScale = Math.round(Math.min(Math.max(0.5, zoomScale + delta), 4.0) * 100) / 100;
  if (zoomScale <= 1) {
    panX = 0;
    panY = 0;
  }
  applyTransform();
  updateZoomUI();
}

function resetZoom() {
  zoomScale = 1;
  panX = 0;
  panY = 0;
  applyTransform();
  updateZoomUI();
}

function updateZoomUI() {
  const resetBtn = document.getElementById('resetZoomBtn');
  if (resetBtn) {
    resetBtn.textContent = zoomScale === 1 ? '1:1' : `${Math.round(zoomScale * 100)}%`;
  }
  if (viewportStage) {
    if (zoomScale > 1) {
      viewportStage.classList.add('can-pan');
    } else {
      viewportStage.classList.remove('can-pan');
    }
  }
}

function applyTransform() {
  if (stageContainer) {
    stageContainer.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomScale})`;
  }
}

function exportJsonData() {
  if (!currentDetectionResult) return;
  const jsonStr = JSON.stringify(currentDetectionResult, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.download = `PCB_Defect_Report_${Date.now()}.json`;
  link.href = url;
  link.click();
  URL.revokeObjectURL(url);
}

async function loadCalibrationStatus() {
  try {
    const res = await fetch('/api/calibration/status');
    if (!res.ok) return;
    const data = await res.json();
    updateCalibrationUI(data);
  } catch (err) {
    console.warn('Failed to load calibration status:', err);
  }
}

function updateCalibrationUI(data) {
  if (!calibStatusText) return;

  if (data && data.is_calibrated) {
    calibStatusText.textContent = `✅ บันทึกแล้ว (${data.reference_features || 0} features)`;
    calibStatusText.className = 'calib-status';
    calibStatusText.style.display = 'block';
    if (calibPreviewRow) calibPreviewRow.style.display = 'flex';
    if (calibInfoLabel) calibInfoLabel.textContent = '✅ ตั้งค่าแล้ว';
    if (calibInfoDate) calibInfoDate.textContent = data.timestamp ? new Date(data.timestamp).toLocaleString('th-TH') : '';
    if (calibRefThumb && data.thumbnail) {
      calibRefThumb.src = data.thumbnail;
    }
    if (clearCalibBtn) clearCalibBtn.style.display = 'block';
  } else {
    calibStatusText.textContent = '';
    calibStatusText.style.display = 'none';
    if (calibPreviewRow) calibPreviewRow.style.display = 'none';
    if (clearCalibBtn) clearCalibBtn.style.display = 'none';
  }
}

function syncCanvasSize() {
  if (!calibrationCanvas || !cameraStreamImg) return;
  const rect = cameraStreamImg.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    calibrationCanvas.width = rect.width;
    calibrationCanvas.height = rect.height;
    calibrationCanvas.style.width = `${rect.width}px`;
    calibrationCanvas.style.height = `${rect.height}px`;
  }
}

function enterCalibrationMode() {
  if (!isCameraActive) {
    alert('กรุณาเปิดกล้องก่อนเริ่ม Calibrate');
    return;
  }

  isCalibrationMode = true;
  calibrationCorners = [];

  syncCanvasSize();
  calibrationCanvas.style.display = 'block';
  calibrationCanvas.classList.add('marking-active');
  if (calibBanner) calibBanner.style.display = 'flex';
  if (calibStatusText) calibStatusText.style.display = 'block';

  startCalibBtn.style.display = 'none';
  calibActionsRow.style.display = 'flex';
  saveCalibBtn.disabled = true;

  updateCalibrationInstruction();
  drawCalibrationOverlay();
}

function exitCalibrationMode() {
  isCalibrationMode = false;
  calibrationCorners = [];

  if (calibrationCanvas) {
    calibrationCanvas.style.display = 'none';
    calibrationCanvas.classList.remove('marking-active');
    if (calibCtx) calibCtx.clearRect(0, 0, calibrationCanvas.width, calibrationCanvas.height);
  }

  if (calibBanner) calibBanner.style.display = 'none';

  startCalibBtn.style.display = 'block';
  calibActionsRow.style.display = 'none';
  loadCalibrationStatus();
}

function updateCalibrationInstruction() {
  const count = calibrationCorners.length;
  let text = '';
  if (count < 4) {
    text = `คลิกจุดที่ ${count + 1}/4: ${CALIB_LABELS[count]}`;
    calibStatusText.textContent = text;
    calibStatusText.className = 'calib-status active-marking';
    saveCalibBtn.disabled = true;
  } else {
    text = '🎉 ครบ 4 มุมแล้ว! ตรวจสอบแล้วกด "บันทึก"';
    calibStatusText.textContent = text;
    calibStatusText.className = 'calib-status';
    saveCalibBtn.disabled = false;
  }
  if (calibBannerText) calibBannerText.textContent = text;
}

function handleCalibrationCanvasClick(e) {
  if (!isCalibrationMode) return;
  if (calibrationCorners.length >= 4) return;

  
  syncCanvasSize();

  const rect = calibrationCanvas.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  
  const scaleX = 1280 / rect.width;
  const scaleY = 720 / rect.height;
  const rawX = Math.round(Math.max(0, Math.min(1280, clickX * scaleX)));
  const rawY = Math.round(Math.max(0, Math.min(720, clickY * scaleY)));

  calibrationCorners.push({
    display: [clickX, clickY],
    raw: [rawX, rawY]
  });

  updateCalibrationInstruction();
  drawCalibrationOverlay();
}

function resetCalibrationCorners() {
  calibrationCorners = [];
  updateCalibrationInstruction();
  drawCalibrationOverlay();
}

function drawCalibrationOverlay() {
  if (!calibCtx || !calibrationCanvas) return;
  calibCtx.clearRect(0, 0, calibrationCanvas.width, calibrationCanvas.height);

  const n = calibrationCorners.length;
  if (n === 0) return;

  
  if (n > 1) {
    calibCtx.beginPath();
    calibCtx.moveTo(calibrationCorners[0].display[0], calibrationCorners[0].display[1]);
    for (let i = 1; i < n; i++) {
      calibCtx.lineTo(calibrationCorners[i].display[0], calibrationCorners[i].display[1]);
    }
    if (n === 4) {
      calibCtx.closePath();
      calibCtx.fillStyle = 'rgba(16, 185, 129, 0.15)';
      calibCtx.fill();
    }
    calibCtx.strokeStyle = n === 4 ? '#10b981' : '#06b6d4';
    calibCtx.lineWidth = 2;
    calibCtx.stroke();
  }

  
  calibrationCorners.forEach((pt, idx) => {
    const [x, y] = pt.display;
    const color = CALIB_COLORS[idx];

    
    calibCtx.beginPath();
    calibCtx.arc(x, y, 9, 0, Math.PI * 2);
    calibCtx.fillStyle = color;
    calibCtx.shadowColor = color;
    calibCtx.shadowBlur = 10;
    calibCtx.fill();
    calibCtx.shadowBlur = 0;

    
    calibCtx.beginPath();
    calibCtx.arc(x, y, 3, 0, Math.PI * 2);
    calibCtx.fillStyle = '#ffffff';
    calibCtx.fill();

    
    calibCtx.font = 'bold 12px "JetBrains Mono", sans-serif';
    calibCtx.fillStyle = '#ffffff';
    calibCtx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
    calibCtx.lineWidth = 3;
    const label = `${idx + 1}. ${CALIB_LABELS[idx]}`;
    calibCtx.strokeText(label, x + 14, y + 4);
    calibCtx.fillText(label, x + 14, y + 4);
  });
}

async function saveCalibration() {
  if (calibrationCorners.length !== 4) {
    alert('กรุณาคลิกให้ครบทั้ง 4 มุมก่อนบันทึก');
    return;
  }

  const payload = {
    corners: calibrationCorners.map(c => c.raw)
  };

  try {
    saveCalibBtn.disabled = true;
    saveCalibBtn.textContent = '⏳ กำลังบันทึก...';

    const res = await fetch('/api/calibration/save_corners', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || data.message || 'บันทึกไม่สำเร็จ');
    }

    alert('✅ Calibrate บอร์ดสำเร็จ!\nระบบจะจำตำแหน่งบอร์ดนี้ทันที');
    exitCalibrationMode();
    await loadCalibrationStatus();

  } catch (err) {
    alert('เกิดข้อผิดพลาดในการบันทึก: ' + err.message);
    saveCalibBtn.disabled = false;
    saveCalibBtn.textContent = '💾 บันทึก';
  }
}

async function clearCalibration() {
  if (!confirm('คุณต้องการลบข้อมูล Calibration ทั้งหมดใช่หรือไม่?')) return;

  try {
    const res = await fetch('/api/calibration/clear', { method: 'POST' });
    if (res.ok) {
      alert('ลบข้อมูล Calibration เรียบร้อย');
      await loadCalibrationStatus();
    }
  } catch (err) {
    alert('เกิดข้อผิดพลาดในการลบ: ' + err.message);
  }
}

function initTheme() {
  const savedTheme = localStorage.getItem('pcb_theme') || 'light';
  applyTheme(savedTheme);

  const themeToggleBtn = document.getElementById('themeToggleBtn');
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      const isDark = document.body.classList.contains('dark-theme');
      const newTheme = isDark ? 'light' : 'dark';
      applyTheme(newTheme);
      localStorage.setItem('pcb_theme', newTheme);
    });
  }
}

function applyTheme(theme) {
  const themeToggleIcon = document.getElementById('themeToggleIcon');
  const themeToggleText = document.getElementById('themeToggleText');
  if (theme === 'dark') {
    document.body.classList.add('dark-theme');
    if (themeToggleIcon) themeToggleIcon.textContent = '☀️';
    if (themeToggleText) themeToggleText.textContent = 'โหมดสว่าง';
  } else {
    document.body.classList.remove('dark-theme');
    if (themeToggleIcon) themeToggleIcon.textContent = '🌙';
    if (themeToggleText) themeToggleText.textContent = 'โหมดมืด';
  }
}
