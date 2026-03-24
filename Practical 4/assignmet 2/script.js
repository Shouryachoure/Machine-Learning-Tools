<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Assignment 3 – MobileNet vs EfficientNet</title>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet"></script>
  <link rel="stylesheet" href="style.css"/>
</head>
<body>
  <header>
    <a href="../../index.html" class="back">← Practical 4</a>
    <div class="tag">Assignment 03</div>
    <h1>MobileNet vs EfficientNet</h1>
    <p>Compare predictions between two pre-trained models on the same image · CO3 – Apply</p>
  </header>

  <div class="upload-section">
    <div class="upload-box" onclick="document.getElementById('cmpInput').click()">
      <div id="upPrompt">
        <div style="font-size:32px">🖼</div>
        <div class="up-text">Click to upload image</div>
        <div class="up-sub">JPG · PNG · WEBP</div>
      </div>
      <img id="cmpImg" style="display:none; max-width:100%; max-height:260px; object-fit:contain;"/>
      <input type="file" id="cmpInput" accept="image/*" hidden onchange="loadImg(event)"/>
    </div>
    <div class="upload-actions">
      <button class="btn-primary" id="cmpBtn" onclick="compareModels()" disabled>⚡ Compare Models</button>
      <button class="btn-outline" onclick="resetCmp()">↺ Reset</button>
    </div>
  </div>

  <!-- Comparison Grid -->
  <div class="compare-grid">
    <div class="model-card" id="cardA">
      <div class="model-header mobilenet">
        <span class="model-badge">Model A</span>
        <span class="model-name">MobileNet v2</span>
        <span class="model-tag">Fast · Lightweight</span>
      </div>
      <div id="resA" class="model-results">
        <div class="model-waiting">Run comparison to see results</div>
      </div>
      <div class="model-stats">
        <div class="mstat"><div id="timeA" class="mstat-val">--</div><div class="mstat-key">ms</div></div>
        <div class="mstat"><div id="confA" class="mstat-val">--</div><div class="mstat-key">Top Conf %</div></div>
      </div>
    </div>

    <div class="vs-divider">VS</div>

    <div class="model-card" id="cardB">
      <div class="model-header efficientnet">
        <span class="model-badge">Model B</span>
        <span class="model-name">MobileNet v1</span>
        <span class="model-tag">Classic · Stable</span>
      </div>
      <div id="resB" class="model-results">
        <div class="model-waiting">Run comparison to see results</div>
      </div>
      <div class="model-stats">
        <div class="mstat"><div id="timeB" class="mstat-val">--</div><div class="mstat-key">ms</div></div>
        <div class="mstat"><div id="confB" class="mstat-val">--</div><div class="mstat-key">Top Conf %</div></div>
      </div>
    </div>
  </div>

  <!-- Verdict -->
  <div class="verdict-card" id="verdictCard" style="display:none">
    <div class="verdict-title">⚖ Comparison Verdict</div>
    <div id="verdictText" class="verdict-text"></div>
  </div>

  <div class="log-box" id="log"><div class="log-line">Ready.</div></div>

  <script src="script.js"></script>
</body>
</html>
