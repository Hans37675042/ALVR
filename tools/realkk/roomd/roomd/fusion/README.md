# roomd.fusion

Quest 環境深度（MSG_DEPTH_FRAME_V2）融合成房間幾何：dense 投影式 TSDF（NVIDIA Warp，GPU），
輸出 MESH_CHUNK、NAV_HEIGHTMAP，並提供查詢 API 給語意層。對應 plan v4 切片 S2（解碼）、
S3（TSDF）、S4（MC／高度圖）、S7（場景先驗）。介面契約以 `docs/CONTRACT-roomd.md` 為準。

## 依賴

- `warp-lang==1.18.0`（Windows CUDA wheel，5090 為 sm_120）、`numpy`、`lz4`
- 測試另需 `pytest`

```powershell
cd tools\realkk\roomd
uv run --no-project --with warp-lang==1.18.0 --with numpy --with lz4 --with pytest python -m pytest tests/fusion -s
```

## 座標

- 輸入 pose 是 OpenXR STAGE（右手系）；所有輸出與查詢都在 **Unity 左手 stage**：位置 `(x, y, -z)`，
  四元數 `(-qx, -qy, qz, qw)`。
- 體素 `(i, j, k)` 的中心在 `origin + (i, j, k) * voxel_size`；origin 對齊 chunk 格線，
  所以 MESH_CHUNK 的 `ix, iy, iz` 是全域 chunk 索引（chunk 涵蓋 `[ix·size, (ix+1)·size)`）。
- 體積在第一個有效幀（或先驗網格）時配置：xz 以頭部（兩個 depth view 位置的中點）為中心，
  y 從 `floor_y − floor_margin` 到 `floor_y + height_above_floor`，各軸向外取整到 chunk。
  預設 8×4×8 個 1 m chunk，2 cm 體素，3200 萬體素，約 0.4 GB VRAM。

## 接到 roomd（FusionSink）

```powershell
uv run python -m roomd --fusion roomd.fusion:create_sink
```

`roomd/fusion/sink.py` 的 `TsdfFusionSink` 實作 roomd-io 的 `roomd.sink.FusionSink`：

| 方法 | 行為 |
|---|---|
| `integrate(protocol.DepthFrame)` | 取 `views[i].pose_xr`／`intrinsics` 與 `view_z(i)` 組成 DepthFrame 後整合（同步，單幀約 3–4 ms） |
| `snapshot_outputs()` | `map_revision`、`floor_y`／`floor_rms`（地板擬合）、`nav_heightmap`（最新一份，f16）、有新圖才 `nav_revision += 1`、`mesh_chunks`（這次要送的）；`objects`／`seats`／`walls` 回 None，由語意層負責 |
| `set_scene_prior(ScenePrior)` | 體積還沒建立時用 `floor_y` 當地板；有 GLOBAL_MESH 就以先驗權重寫入 |
| `on_playspace_changed(pose)` | pose 和上次不同（>1 mm 或 >0.1°）就清掉體積：已送過的 chunk 下次以 `vcount=0`、更高 revision 送出移除，高度圖改送全未知並遞增 `nav_revision` |

9945 payload 的編解碼由 `roomd.protocol` 負責；`roomd.fusion.outputs` 只是轉成 protocol 型別的薄包裝。

## 對外 API

```python
from roomd.fusion import TsdfFusion, FusionConfig, Obb

fusion = TsdfFusion(FusionConfig())          # 所有參數見 config.py
fusion.integrate(payload_or_DepthFrame)      # False = 假幀（全 0x8080）已丟棄；格式錯誤 raise ValueError
fusion.integrate_prior_mesh(verts, indices)  # S7：Unity 座標的場景 global mesh，預設權重 2
out = fusion.snapshot_outputs(now=None, force=False)
#   out.mesh_chunks: [MeshChunk]      需要送的 chunk（≤2 Hz/chunk，變化量最大者優先，每次最多 16 個）
#   out.heightmap:   Heightmap | None （≤1 Hz，有新資料才給）
#   out.floor:       FloorPlane | None
#   out.stats:       {"frames","fakeFrames","integrateMsP95","integrateMsP50","mapRevision","voxels"}
```

| 查詢 | 回傳 |
|---|---|
| `floor_plane()` | `FloorPlane(y, normal, rms, count)`：`floor_y ± 0.15 m` 內朝上表面點的穩健平面擬合 |
| `heightmap()` | `Heightmap`：5 cm 格，陣列索引 `[z, x]`；`sample(x, z) -> (floorY, topY, flags)` |
| `surface_points(min_y, max_y, region=None)` | `(points (N,3), normals (N,3))`：朝上表面（法線與 +Y 夾角 ≤ ~32°），region 可給 `Obb` |
| `free_fraction(obb)` | OBB 內「觀測到、在表面前方」的體素比例 |
| `occupied_fraction(obb)` | OBB 內「觀測到、在表面上或後方截斷帶內」的體素比例 |
| `visible_fraction(obb, max_age_frames=10)` | OBB 內最近 N 幀有被更新的體素比例 |
| `region_stats(obb, max_age_frames)` | 上面三者的原始計數（含 `total`） |
| `extract_mesh()` | 整個體積的網格（除錯／離線用） |
| `save_state(path)`／`TsdfFusion.load_state(path)` | 體積與計數器存成 .npz（約 10–15 MB）再讀回，查詢與 snapshot 都能用；給語意層離線試跑 |

- 三種比例的分母都是 OBB 內的全部體素；未觀測（例如物體內部）兩邊都不算。判斷物件消失時，
  建議同時看 `visible_fraction`（有沒有被看到）和 `free_fraction`（看到的是空的）。
- `Obb(center, half_extents, yaw)`：yaw 為繞 +Y 的弧度，Unity 慣例（正值把 +Z 轉向 +X）。

### 9945 payload 約定

- MESH_CHUNK：頂點是**絕對** Unity stage 座標；三角形繞向採 Unity 正面慣例
  （`cross(b−a, c−a)` 指向自由空間）；`vcount = 0` 表示該 chunk 已清空。離地 3 cm 內的三角形已排除。
- NAV_HEIGHTMAP：`originX, originZ` 是格 (0,0) 的最小角；第 `z*w + x` 筆對應格 `(x, z)`。
  `topY = floorY` 表示沒有障礙。flags：bit0 known、bit1 obstacle（離地 5 cm–1.7 m 內有佔用；角色身高 1.5–1.6 m＋餘裕，上舖底面約 1.55 m 擋路、天花板梁不擋）、
  bit2 walkable（known、無障礙、看到地板）。

## 演算法

1. **解碼**（`decode.py`）：D16 → 線性深度（far 有限／inf 兩式），0 與 0xFFFF 視為無效；
   上下疊的兩個 view 拆開；`flip_rows` 可翻轉 row。
2. **遮罩**（GPU）：外圍各裁 10%、量程 0.5–4.5 m、表面法線與視線夾角 > 75° 丟棄（同時去掉深度不連續處的飛點）、
   使用者身體圓柱（頭部下方、水平半徑 0.4 m）內的量測點丟棄、落在 `floor_y − 0.05 m` 以下的量測點丟棄
   （實錄房間往下看時約 40% 的地板樣本落在地板後方 5–30 cm，不擋的話會把地板 carve 掉）。
3. **整合**：每個體素投影到兩個 view，`sdf = (d − z) · |p| / z`；`sdf < −trunc` 不更新，
   其餘 `tsdf = min(1, sdf/trunc)` 做移動平均 `1/(n+1)`、權重上限 16。截斷帶前方全部寫成自由空間（carving）。
   場景變動：體素原本有把握（|tsdf| > 0.5）而新樣本號相反時，權重先乘 0.25 再平均，
   所以家具搬走後舊位置數幀內清掉、搬來的位置也幾乎和「從沒看過」一樣快出現（合成測試：清除 3 幀、出現落後 ≤ 2 幀）。
4. **變化里程表**：每個 1 m chunk 累加 `|Δtsdf|`，超過門檻才重新 MC。
5. **MC**：每個 chunk 帶一格接縫做 marching cubes；只保留 8 角都觀測過、且沒有截斷跳變的格子，
   避免已知／未知邊界與截斷帶背面的假面。
6. **高度圖**：每格掃描體素柱，插值出朝上表面的零交點；接近地板的算地板，地板以上的算障礙。
7. **先驗**：場景 mesh 用 `wp.Mesh` 求帶號距離，只寫截斷帶內，權重 2；實測深度數幀就會蓋過。

## 離線回放

```powershell
uv run --no-project --with warp-lang==1.18.0 --with numpy --with lz4 python -m roomd.fusion.replay D:\AIP\Data\realkk\taps\X.rktap --out OUT
uv run --no-project --with warp-lang==1.18.0 --with numpy --with lz4 python -m roomd.fusion.replay --make-synthetic synth.rktap
```

`--until S` 只融合 tap 開頭 S 秒；`--save-state FILE.npz` 另存體積。輸出 `mesh.ply`、`heightmap.png`（黑＝未知、灰＝可走、紅到黃＝障礙高度，圖上方＝+Z）、`stats.json`。

## 實機資料驗證（2026-10-09 smoke／room tap）

- 解碼公式、row 方向（不翻）、左右 view、Unity 翻轉都對：靜止段地板 y = +4.7 mm、傾斜 1°、RMS 3.9 mm；
  翻 row 會讓地板傾斜 19°，翻 column 兩眼點雲對不上（中位距離 23 cm vs 1.7 cm）；
  地板量測深度與預期深度的比值在影像中央各帶都在 1.00 ± 2%。
- pose 與深度沒有固定時間差（±100 ms 掃描，0 ms 最佳），但轉頭越快幀間越不一致：
  <10°/s 1.5 cm、10–30°/s 2.6 cm、30–60°/s 3.3 cm、>60°/s 5.9 cm。

- room2 tap（129 s，PM 移椅子約 1.36 m）：地板 y = +1.7 cm、RMS 7.5 mm；主牆面偏離垂直 0.4°。
  舊位置第一次重新入鏡後約 0.9 s（4 幀）清空；新位置入鏡後 1 s 達一半、4–6 s 與從未看過的體積同樣完整
  （未加衝突衰減前要 4 倍時間）。坐上新椅子時身體圓柱擋住，椅子沒被侵蝕、人也沒進地圖；之後在桌前的動作沒有在頭前留下殘影。

## 已知限制

- 體積不會跟著使用者移動：超出首幀為中心的 8×8 m 範圍就看不到。
- D16 編碼公式與 row 方向尚未實機驗證（REALKK.md），錯了的話要調 `flip_rows` 或解碼式。
- 先驗 mesh 的繞向假設是「Unity 正面朝自由空間」；Meta global mesh 實際繞向待實機確認。
- 實錄房間的高度圖障礙格偏多（8527 已知格中 7063 是障礙）。主要是 1.5–2.1 m 的水平結構（向下表面集中在 1.55 m 與 1.9–2.1 m，天花板約 2.36 m），
  是不是真的家具（上舖、吊櫃）要 PM 確認；障礙判定門檻屬 S9 實機調參。
- 天花板等無紋理面同樣有「量到表面後方」的樣本，目前只擋地板以下。
- 身體遮罩只看頭部 pose，手伸出 0.4 m 外仍會被整合（client 端 hand removal 是 S8）。
- 效能（5090、2×320×320、3200 萬體素）：單幀整合（含解碼與遮罩）p50 ~2.7 ms、p95 ~4 ms；
  每個 chunk 的 MC 約 9 ms，首次呼叫要多花數百 ms 編譯 kernel。
