# link_spike：Quest Link 開發者模式下 PC 原生 OpenXR 取深度與場景

plan v4 選配 spike（R16）。要回答的問題只有一個：開了 Developer Runtime Features 的
Meta Quest Link PC runtime，能不能讓 PC 上的原生 OpenXR 程式直接拿到
`XR_META_environment_depth` 深度幀與 `XR_FB_scene` 場景資料。

KKS 走 OpenVR，所以就算可行，這條路也只是開發／測試管道；要不要取代 ALVR 另外評估。

## 判讀標準

**可行＝深度 ≥ 9 fps 而且讀得到像素，並且有 scene 資料。** `report.json` 的
`verdict.feasible` 會照這條自動算：

| 欄位 | 意思 |
|---|---|
| `verdict.depth_fps` | 10 秒內「內容有變」的深度幀數換算的 fps（同一張圖重複拿到不算） |
| `verdict.depth_pixels_ok` | D3D11 readback 全程沒有錯 |
| `verdict.scene_ok` | 至少拿到一個 room layout 或一個有 label 的 anchor |
| `verdict.last_failed_step` | 最後失敗的步驟名稱，可對照下面的「失敗點對照表」 |

## 程式做什麼

1. 選 runtime：`--runtime-json` > 環境變數 `XR_RUNTIME_JSON` > 登錄檔 ActiveRuntime。
   程式不經過 Khronos loader，而是自己讀 manifest、載入 runtime DLL、做 negotiation，
   所以**不必改系統的 active runtime**，也不會載入任何 API layer。
2. 列出 runtime 名稱、版本與所有擴充 → `extensions.json`。
3. 建 instance（開 runtime 有提供的 D3D11／headless／depth／scene／mesh 相關擴充）並取 HMD system。
4. 建 session：有 `XR_KHR_D3D11_enable` 就用最小 D3D11 device。深度像素要靠圖形 API 才讀得出來，
   所以 D3D11 優先；只有在沒有 D3D11 但有 `XR_MND_headless` 時才退到 headless，這時只做 scene。
5. 深度（`--seconds`，預設 10 秒）：建 provider／swapchain、start、每幀 acquire，
   D3D11 把兩個 array slice 讀回 CPU，上下堆疊（view 0 在上），以 `MSG_DEPTH_FRAME_V2`
   版面寫進 `depth.rktap`。格式是 RawD16（format 0），值是 runtime 給的原始 D16，跟 fork 一樣不換算。
6. 場景：`xrQuerySpacesFB(ROOM_LAYOUT)` 拿 floor／ceiling／walls uuid，
   再查所有帶 SEMANTIC_LABELS 的 anchor，列 label、bbox2D／3D、STAGE 座標下的 pose，
   有 TRIANGLE_MESH_META 的就取三角形數（GLOBAL_MESH）→ `scene.json`。
7. 每一步都把 XrResult 記進 `report.json`，某步失敗就跳過依賴它的步驟，其他照跑。

## PM 實測步驟

### 0. 事前準備（一次性）

1. 安裝 **Meta Quest Link** PC app（本機目前沒裝，見下方「本機驗證結果」）。
   安裝程式或首次啟動可能會問要不要設成 OpenXR active runtime：**選否**，
   以免影響 ALVR／SteamVR、Virtual Desktop 的設定。萬一被改了，到原本使用的 runtime
   （目前是 Virtual Desktop Streamer）的設定裡設回 active runtime 即可。
2. 頭盔先做過 **空間設定（Space Setup）**，房間掃描完成，scene 才有資料。
3. Link app → 設定 → **Beta**，打開：
   - **Developer Runtime Features**
   - **Passthrough over Meta Quest Link**（舊版叫 Passthrough over Oculus Link）
   - **Spatial Data over Meta Quest Link**
   開關名稱會隨版本變動，以「開發者 runtime」「穿透」「空間資料」三個關鍵字對應。
   打開後如果提示重啟 Link 服務就照做。
4. Build（PowerShell）：

   ```powershell
   . D:\AIP\.tools\alvr-toolchain.ps1
   cd D:\AIP\.wt\alvr-realkk\link-spike\tools\realkk\link_spike
   cargo build --release
   ```

### 1. 執行

1. 先結束 ALVR 串流（頭盔同時只能走 Link 或 ALVR 其中一條）。
2. 接 Link 線（或 Air Link），頭盔進入 Link 的 home 畫面並戴著；在房間裡看向有家具的方向。
3. 執行：

   ```powershell
   cd D:\AIP\.wt\alvr-realkk\link-spike\tools\realkk\link_spike
   $env:XR_RUNTIME_JSON = 'C:\Program Files\Oculus\Support\oculus-runtime\oculus_openxr_64.json'
   .\target\release\link_spike.exe --seconds 10
   Remove-Item Env:XR_RUNTIME_JSON
   ```

   Link 裝在別的位置時，`report.json` 的 `oculus_runtime.candidates` 會列出程式找過的路徑；
   也可以改用 `--runtime-json <路徑>`。
4. 頭盔裡可能跳出「允許存取空間資料」之類的權限提示，按允許後重跑一次。
5. 執行期間螢幕是空的（程式不繪圖），跑完約 10–25 秒自己結束。

### 2. 輸出

都在 `out\<UTC 時間戳>\`（可用 `--out <資料夾>` 指定）：

| 檔案 | 內容 |
|---|---|
| `report.json` | 每步結果、XrResult、session 狀態變化、`verdict` |
| `extensions.json` | runtime 名稱／版本、全部擴充與版本、目標擴充是否存在 |
| `depth_summary.json` | 解析度、texture 描述、acquire 統計、fps、最大間隔、near/far、首末幀 pose／FOV／intrinsics、D16 值域 |
| `depth.rktap` | 深度幀，可直接給 `tap_inspect.py`／融合工具讀 |
| `scene.json` | room layout、每個 anchor 的 label／component／bbox／pose、mesh 三角形數 |

看深度錄影：

```powershell
cd D:\AIP\.wt\alvr-realkk\link-spike\tools\realkk
python tap_inspect.py --no-decode link_spike\out\<時間戳>\depth.rktap
```

（要檢查全 0x80 假幀就 `pip install lz4` 後拿掉 `--no-decode`。）

### 3. 失敗點對照表

| `last_failed_step`／現象 | 最可能原因 |
|---|---|
| `load_runtime`：cannot read runtime manifest | Link app 沒裝或裝在別處，改用 `--runtime-json` |
| `enumerate_extensions` 的 `wanted_present` 裡 `XR_META_environment_depth`／`XR_FB_scene` 是 false | Developer Runtime Features 沒開，或 Link 服務沒重啟 |
| `get_system(HMD)`：`ERROR_FORM_FACTOR_UNAVAILABLE` | 頭盔沒連上 Link 或沒戴著 |
| `session_running` 失敗、states 停在 IDLE | Link 沒進 home、頭盔休眠，或別的 VR app 佔著 |
| `depth_create` 某步失敗 | 看該步的 XrResult；`environment_depth_props.supports_environment_depth` 為 false 表示 runtime 不支援 |
| `depth_capture` 的 acquire 全是 not_available | Passthrough over Link 沒開，或缺權限 |
| `readback_errors` 有內容 | runtime 給的 texture 格式／版面跟預期不同；附上 `depth_summary.json` 的 `texture_desc` 回報 |
| `fps` < 9 | Link 頻寬或 dev runtime 限制，記錄 `max_gap_ms` |
| `scene` 的 `anchor_count` 為 0 | Spatial Data over Link 沒開、沒做 Space Setup，或空間資料權限被拒 |

## 已知假設與限制

- **列順序**：D3D11 texture 第 0 列就是影像最上面一列，所以沒有像 fork client（GLES readback）那樣上下翻轉。
  第一次實測時拿 `tap_inspect`／融合工具和 ALVR 錄的 tap 比對一下方向，反了就是這條假設錯了。
- **座標原點**：pose 在 PC runtime 的 STAGE（取不到則 LOCAL）空間，與 ALVR 重定原點後的 stage 不同，
  跨管道比對不能假設同一個原點。
- **時間**：`client_timestamp_ns` 是 predicted display time，不是深度實際擷取時間；
  有 `XR_KHR_win32_convert_performance_counter_time` 時會算出 unix−XrTime 的 offset 填進 header。
- **新幀判定**：用像素內容的 hash；同一張圖被重複 acquire 不算新幀。
- 程式沒有開 passthrough layer。如果 acquire 一直 not_available 而開關都對，下一步是加上
  `XR_FB_passthrough` 啟動後再試（目前不在 spike 範圍）。
- 沒做 hand removal、沒觸發 Space Setup（`XR_FB_scene_capture`）。

## 其他用法

- `--list-only`：只到列擴充為止，不建 instance／session，可以安全地拿來看任何 runtime 有哪些擴充。
- `--no-depth`／`--no-scene`：只跑其中一項。
- `--synth-tap <檔案>`：不碰 OpenXR，產生 30 幀合成深度 tap，用來確認 rktap 相容性。
- 測試：`cargo test`（`tap_layout` 鎖定 rktap 與 depth V2 header 位元組版面；
  `d3d_readback` 在本機 GPU 上驗證兩層 D16 texture array 的 readback）。

## 本機驗證結果（2026-10-09，沒接頭盔）

- Meta Quest Link app **沒有安裝**：`C:\Program Files\Oculus` 不存在、沒有 `OculusBase` 環境變數、
  沒有 OVRService 服務；登錄檔 AvailableRuntimes 只有 SteamVR 與 Virtual Desktop，
  ActiveRuntime 是 Virtual Desktop。
- `XR_RUNTIME_JSON` 指向 Oculus 預設路徑執行：`select_runtime` OK → `load_runtime` 失敗
  （找不到 manifest），`verdict.last_failed_step = load_runtime`，符合預期。
- 為了驗證自製的 runtime 載入／negotiation 與擴充列舉，用 `--list-only --runtime-json`
  對 Virtual Desktop runtime 跑過一次（不建 instance）：negotiation 成功（runtime API 1.0.34），
  列出 30 個擴充；VD 沒有 depth／scene 類擴充，這是預期中的。SteamVR runtime 刻意沒碰。
- `cargo test`：`tap_layout` 3 項、`d3d_readback` 1 項全過（後者用 D16_UNORM depth-stencil texture array）。
- `--synth-tap` 產出的檔案用 `tap_inspect.py` 讀取正常，`depth_listener.decode_d16` 解出 640×320。
- instance 之後的步驟（system、session、frame loop、depth acquire、scene query）要接頭盔才驗得到。
