(() => {
  const input = document.getElementById('image-file');
  const dropzone = document.getElementById('dropzone');
  const previewWrap = document.getElementById('preview-wrap');
  const previewImage = document.getElementById('preview-image');
  const fileName = document.getElementById('file-name');
  const analyzeButton = document.getElementById('analyze-button');
  const message = document.getElementById('form-message');
  const resultEmpty = document.getElementById('result-empty');
  const resultData = document.getElementById('result-data');
  const resultState = document.getElementById('result-state');
  const menuButton = document.querySelector('.menu-toggle');
  const nav = document.getElementById('navigation');
  const stoneThreshold = document.getElementById('stone-threshold');
  const modelStatus = document.getElementById('model-status');
  let selectedFile = null;
  let previewUrl = null;
  updateModelStatus();

  menuButton.addEventListener('click', () => {
    const open = menuButton.getAttribute('aria-expanded') !== 'true';
    menuButton.setAttribute('aria-expanded', String(open));
    menuButton.setAttribute('aria-label', open ? 'Tutup navigasi' : 'Buka navigasi');
    nav.classList.toggle('open', open);
  });
  nav.querySelectorAll('a').forEach(link => link.addEventListener('click', () => {
    nav.classList.remove('open'); menuButton.setAttribute('aria-expanded', 'false'); menuButton.setAttribute('aria-label', 'Buka navigasi');
  }));
  dropzone.addEventListener('keydown', event => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => { if (input.files[0]) setFile(input.files[0]); });
  ['dragenter', 'dragover'].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.add('dragging'); }));
  ['dragleave', 'drop'].forEach(type => dropzone.addEventListener(type, event => { event.preventDefault(); dropzone.classList.remove('dragging'); }));
  dropzone.addEventListener('drop', event => { const file = event.dataTransfer.files[0]; if (file) setFile(file); });
  document.getElementById('remove-file').addEventListener('click', clearFile);
  stoneThreshold.addEventListener('input', () => { document.getElementById('stone-threshold-value').textContent = `${stoneThreshold.value}%`; });

  function setFile(file) {
    message.textContent = '';
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['png', 'jpg', 'jpeg'].includes(ext) || !['image/png', 'image/jpeg'].includes(file.type)) {
      message.textContent = 'Format file tidak didukung. Pilih JPG, JPEG, atau PNG.'; return;
    }
    if (file.size > 16 * 1024 * 1024) { message.textContent = 'Ukuran file maksimal 16MB.'; return; }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    previewImage.src = previewUrl;
    fileName.textContent = file.name;
    dropzone.hidden = true; previewWrap.hidden = false; analyzeButton.disabled = false;
    resultEmpty.hidden = false; resultData.hidden = true; resultState.textContent = 'MENUNGGU ANALISIS';
    resultState.dataset.state = 'empty';
  }
  function clearFile() {
    selectedFile = null; input.value = ''; previewWrap.hidden = true; dropzone.hidden = false; analyzeButton.disabled = true;
    message.textContent = ''; if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null;
  }
  analyzeButton.addEventListener('click', async () => {
    if (!selectedFile) return;
    analyzeButton.disabled = true; analyzeButton.textContent = 'Memproses gambar...';
    message.className = 'form-message'; message.textContent = 'Model sedang menganalisis CT Scan Anda.';
    resultState.textContent = 'SEDANG DIPROSES'; resultState.dataset.state = 'loading'; resultEmpty.hidden = false; resultData.hidden = true;
    try {
      const body = new FormData();
      body.append('file', selectedFile);
      body.append('stone_confidence', String(Number(stoneThreshold.value) / 100));
      const response = await fetch('/api/detect', { method: 'POST', body });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.message || 'Analisis tidak berhasil.');
      showResults(data);
      message.textContent = 'Analisis selesai. Tinjau hasil bersama tenaga medis profesional.';
      message.style.color = '#426b4d';
    } catch (error) {
      resultState.textContent = 'ANALISIS GAGAL'; resultState.dataset.state = 'error';
      message.style.color = '#9b3c42'; message.textContent = error.message || 'Tidak dapat menghubungi server. Coba kembali.';
    } finally {
      analyzeButton.disabled = false; analyzeButton.textContent = 'Mulai analisis';
    }
  });
  function showResults(data) {
    resultEmpty.hidden = true; resultData.hidden = false; resultState.textContent = 'ANALISIS SELESAI'; resultState.dataset.state = 'done';
    const image = document.getElementById('result-image'); image.src = data.result_image;
    const stones = data.detections.filter(item => /stone|batu/i.test(item.class)).length;
    const kidneys = data.detections.filter(item => /kidney|ginjal/i.test(item.class)).length;
    const status = document.getElementById('result-status');
    status.textContent = stones ? `${stones} kandidat batu, ${kidneys} ginjal ditandai` : `${kidneys} ginjal ditandai, tidak ada kandidat batu di atas ambang`;
    document.getElementById('result-count').textContent = `${data.total} objek ditemukan`;
    const summary = document.getElementById('class-summary'); summary.replaceChildren();
    [['Ginjal', kidneys], ['Kandidat batu', stones]].forEach(([label, count]) => {
      const item = document.createElement('div'); item.className = 'class-count';
      const name = document.createElement('span'); name.textContent = label;
      const value = document.createElement('strong'); value.textContent = count;
      item.append(name, value); summary.append(item);
    });
    const list = document.getElementById('detection-list'); list.replaceChildren();
    if (!data.detections.length) {
      const row = document.createElement('p'); row.className = 'form-message'; row.style.color = '#625b69'; row.textContent = `Tidak ada objek di atas ambang Kidney ${Math.round(data.thresholds.kidney * 100)}% dan Stone ${Math.round(data.thresholds.stone * 100)}%. Citra tetap dapat diperiksa dengan radiolog.`; list.append(row);
    }
    data.detections.forEach(item => {
      const row = document.createElement('div'); row.className = 'detection-row';
      const label = document.createElement('strong'); label.textContent = item.class;
      const confidence = document.createElement('span'); confidence.textContent = `${Math.round(item.confidence * 100)}%`;
      const bbox = document.createElement('span'); bbox.textContent = item.bbox.map(n => Math.round(n)).join(', '); bbox.title = 'Koordinat bounding box: x1, y1, x2, y2';
      row.append(label, confidence, bbox); list.append(row);
    });
    document.getElementById('download-result').href = data.result_image;
  }
  function updateModelStatus() {
    const items = [...modelStatus.querySelectorAll('.model-pill')];
    items.forEach(item => {
      item.classList.add('active');
      if (item.textContent.startsWith('Kidney')) item.textContent = 'Kidney · mask CT';
      if (item.textContent.startsWith('Stone')) item.textContent = 'Stone · pseudo-label';
    });
  }
})();
