# CONTEXT.md

## 0. 瀵硅瘽涓庤〃杈捐鍒?
- 榛樿鎶婂洖绛斿璞¤涓衡€滀粠闆跺紑濮嬩簡瑙ｉ」鐩殑浜衡€濄€?- 瑙ｉ噴姒傚康鏃朵紭鍏堜娇鐢ㄤ腑鏂囥€?- 闄ら潪蹇呴』锛屼笉瑕佷腑鑻卞す鏉傘€?- 濡傛灉蹇呴』浣跨敤鑻辨枃鏈锛岃绔嬪埢鍦ㄥ悗闈㈣ˉ涓枃閲婁箟锛屼緥濡傗€滃钩鍧囩粷瀵硅宸紙Mean Absolute Error锛孧AE锛夆€濄€?- 璁插浘銆佽鎸囨爣銆佽妯″瀷鏃讹紝鍏堣鈥滃畠鏄粈涔堚€濓紝鍐嶈鈥滃畠璇存槑浜嗕粈涔堚€濓紝鏈€鍚庡啀璇粹€滄€庝箞瑙ｈ鈥濄€?- 娑夊強瀹规槗娣锋穯鐨勬湳璇椂锛屼紭鍏堢粰鍑虹洿鐧借В閲婏紝涓嶉粯璁ゅ亣璁捐鑰呭凡鏈夋満鍣ㄥ涔犳垨鎺у埗鑳屾櫙銆?- forecasting 榛樿缁撴灉鍥惧彧淇濈暀涓夌被锛?  - `forecast_examples`锛氬崟娆￠娴嬫牱渚嬪浘
  - `forecast_rollout`锛氭粴鍔ㄥ绐楅娴嬪浘
  - `horizon_mae`锛氶娴嬫闀胯宸浘
- `forecast_error_heatmap` 宸茬Щ闄わ紝涓嶅啀浣滀负榛樿缁撴灉鍥捐緭鍑恒€?- `forecast_first_step_rollout` 涔熷凡绉婚櫎锛屼笉鍐嶄綔涓洪粯璁ょ粨鏋滃浘杈撳嚭銆?
## 1. 浣跨敤鏂瑰紡

杩欐槸鏈」鐩殑闀挎湡涓婁笅鏂囨枃浠躲€?
寤鸿瑙勫垯锛?
1. 姣忔寮€鍚柊瀵硅瘽鏃讹紝浼樺厛璇诲彇鏈枃浠躲€?2. 濡傛灉浣跨敤鏀寔鏂囦欢涓婁笅鏂囩殑 IDE / AI 鍔╂墜锛岀洿鎺ュ紩鐢ㄦ湰鏂囦欢銆?3. 姣忔瀹屾垚鏈夋剰涔夌殑浠ｇ爜鏀瑰姩銆佸疄楠岀粨鏋滄洿鏂般€佽矾绾胯皟鏁村悗锛岄兘瑕佹洿鏂版湰鏂囦欢銆?4. 鏈枃浠朵紭鍏堣褰曠ǔ瀹氫簨瀹炪€佸綋鍓嶄富绾裤€佸叧閿喅绛栥€佹渶鏂扮粨鏋溿€乀ODO 鍜屽伐浣滆鍒欍€?

## 2. 椤圭洰褰撳墠涓荤嚎

褰撳墠鐩爣涓嶆槸澶嶇幇鍘熻崏鑾撹鏂囷紝鑰屾槸鍋氾細

**闈㈠悜鎺у埗鐨勬俯瀹ゅ姝ラ娴?+ 闂幆 MPC**

鏍稿績璁惧畾锛?
- 浣跨敤澶氬彉閲忔俯瀹ゆ暟鎹?- 杈撳叆鍘嗗彶瀹ゅ唴鐘舵€?- 杈撳叆鏈潵澶╂皵 / 澶栫敓閲?- 杈撳叆鏈潵鎺у埗璁惧畾鍊?- 棰勬祴鏈潵瀹ゅ唴娓╁鐘舵€?- 鏈€缁堟湇鍔′簬 MPC
- SAC 浠呬綔涓?baseline锛屼笉鏄富绾挎柟娉?

## 3. 褰撳墠椤圭洰鍒嗗伐

### 3.1 鏃ч」鐩?
- [diffmpc](c:/repositories/strawberry/diffmpc)

璇存槑锛?
- 杩欐槸鏃ц崏鑾撲富绾块」鐩?- 涓嶅啀浣滀负鏂扮殑涓昏钀藉湴鏂瑰悜
- 浠呬繚鐣欏弬鑰冧环鍊?
### 3.2 鏂颁富椤圭洰

- [agc_mpc](c:/repositories/strawberry/agc_mpc)

璇存槑锛?
- 杩欐槸鏂扮殑 AGC 2019 涓荤嚎椤圭洰
- 鍚庣画涓昏浠ｇ爜宸ヤ綔閮藉簲浼樺厛鏀惧湪杩欓噷
- 鍘熷垯涓婁笉瑕佸啀鎶婃柊寮€鍙戠户缁爢鍥?`diffmpc`


## 4. 褰撳墠鏍稿績鏁版嵁闆?
### 4.1 涓绘暟鎹泦

- [AutonomousGreenhouseChallenge_edition2](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2)

杩欐槸褰撳墠涓诲缓妯℃暟鎹簮銆?
### 4.2 鍘熷鍖?/ 澶囦唤

- [Autonomous Greenhouse Challenge, Second Edition (2019)_1_all](c:/repositories/strawberry/Autonomous%20Greenhouse%20Challenge,%20Second%20Edition%20(2019)_1_all)

璇存槑锛?
- 杩欐槸鍘熷涓嬭浇鍖呭強鍏惰В鍘嬪悗鐨勫浠界粨鏋?- 涓嶄綔涓轰富寤烘ā鍏ュ彛
- 浠呭湪闇€瑕佸洖鏌ュ師濮嬫牸寮忔椂浣跨敤

### 4.3 鑽夎帗鏁版嵁

- [Strawberry Greenhouse Environmental Control Dataset(version2).csv](c:/repositories/strawberry/Strawberry%20Greenhouse%20Environmental%20Control%20Dataset(version2).csv)

璇存槑锛?
- 鐜板湪闄嶇骇涓?secondary dataset / stress test
- 涓嶅啀浣滀负璁烘枃涓诲疄楠屾暟鎹泦


## 5. AGC 鏁版嵁鐞嗚В

鏉冨▉鍙傝€冿細

- [ReadMe.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/ReadMe.pdf)
- [Economics.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/Economics.pdf)
- [AGC_DATA_SCHEMA.md](c:/repositories/strawberry/AGC_DATA_SCHEMA.md)

鍏抽敭缁撹锛?
- `Weather.csv` = 鏈潵澶栫敓澶╂皵
- `GreenhouseClimate.csv` = 瀹ゅ唴鐘舵€?+ 鎵ц鍣ㄧ姸鎬?+ 璁惧畾鍊?- `*_sp` = 璇锋眰璁惧畾鍊?- `*_vip` = realized setpoints
- `Resources.csv` = 鏃ュ昂搴﹁祫婧愭秷鑰?- `Production.csv` = 鏀惰幏鏃剁偣浜ч噺
- `TomQuality.csv` = 鍝佽川
- `GrodanSens.csv` = 鏍瑰尯 / 鍩鸿川鏁版嵁

褰撳墠绗竴鐗堝缓妯℃帴鍙ｏ細

- `x_past`: 鍘嗗彶瀹ゅ唴鐘舵€?+ 鎵ц鍣ㄥ弽棣?- `w_future`: 澶╂皵 + 鏃堕棿鐗瑰緛
- `u_future`: 鏈潵 setpoints
- `y_future`: 鏈潵 `Tair / Rhair / CO2air / Tot_PAR`


## 6. 褰撳墠鏂板伐绋嬩唬鐮佺姸鎬?
椤圭洰鐩綍锛?
- [agc_mpc](c:/repositories/strawberry/agc_mpc)

鏍稿績鏂囦欢锛?
- [config.py](c:/repositories/strawberry/agc_mpc/config.py)
- [schema.py](c:/repositories/strawberry/agc_mpc/schema.py)
- [processor.py](c:/repositories/strawberry/agc_mpc/data_processing/processor.py)
- [gru_forecaster.py](c:/repositories/strawberry/agc_mpc/models/gru_forecaster.py)
- [dlinear_forecaster.py](c:/repositories/strawberry/agc_mpc/models/dlinear_forecaster.py)
- [seg_rnn_forecaster.py](c:/repositories/strawberry/agc_mpc/models/seg_rnn_forecaster.py)
- [transformer_forecaster.py](c:/repositories/strawberry/agc_mpc/models/transformer_forecaster.py)
- [transformer_hybrid_forecaster.py](c:/repositories/strawberry/agc_mpc/models/transformer_hybrid_forecaster.py)
- [hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/models/hybrid_residual_forecaster.py)
- [trainer.py](c:/repositories/strawberry/agc_mpc/training/trainer.py)
- [evaluator.py](c:/repositories/strawberry/agc_mpc/evaluation/evaluator.py)
- [main.py](c:/repositories/strawberry/agc_mpc/main.py)
- [benchmark_hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/benchmark_hybrid_residual_forecaster.py)
- [README.md](c:/repositories/strawberry/agc_mpc/README.md)

褰撳墠宸插畬鎴愶細

- AGC 鏁版嵁璇诲彇
- 鏃堕棿瀛楁鏍囧噯鍖?- 澶╂皵涓庢皵鍊欒〃瀵归綈
- `sp/vip` 缂哄け鍥炲～
- `x_past / w_future / u_future / y_future` 鏍锋湰鍒囩墖
- 鍗曢殧闂村拰澶氶殧闂存敮鎸?- 鍏ㄥ眬 leak-free 鏃跺簭鍒囧垎
- 澶氶殧闂磋仈鍚堣缁冧笅鐨勫叏灞€鏍囧噯鍖?- 鏉′欢 GRU baseline
- 鏉′欢 DLinear baseline
- 鏉′欢 SegRNN baseline
- 鏉′欢绾?Transformer baseline
- 鏉′欢 Transformer-hybrid baseline
- `DLinear main path + Transformer-hybrid residual` 娣峰悎娈嬪樊妯″瀷鍘熷瀷
- 绂荤嚎璇勪及杈撳嚭
- forecast 鍥炬敮鎸佲€滃巻鍙蹭笂涓嬫枃 + 鏈潵 horizon鈥濊仈鍚堝睍绀猴紝涓嶅啀鍙洴鐫€绾?future window
- forecast 鍥炬柊澧?rolling multi-step rollout 灞曠ず锛岀敤鏇撮暱鏃堕棿杞存樉绀鸿繛缁绐楅娴嬶紝鑰屼笉鍙槸涓€娈?24-step future window
- forecasting 鐜板湪榛樿淇濈暀 3 绫婚娴嬪浘锛氬崟娆￠娴嬫牱渚嬪浘銆佹粴鍔ㄥ绐楅娴嬪浘銆侀娴嬫闀胯宸浘
- 涓婅堪 3 绫婚暱鏃堕棿杞村浘宸茬粡涓?`GRU / DLinear / SegRNN / Transformer / Transformer-hybrid` 鍏ㄩ儴琛ラ綈
- `results` 鐩綍寮€濮嬫寜 `forecasting / control` 鍒嗗眰鏁寸悊
- forecasting checkpoint 缁熶竴鏀舵暃鍒?`agc_mpc/results/forecasting/checkpoints`
- forecasting 鍥剧粺涓€鏀舵暃鍒?`agc_mpc/results/forecasting/figures`
- control summary 缁熶竴鏀舵暃鍒?`agc_mpc/results/control/summaries`
- AGC 鎺у埗渚у垵鐗堟帴鍏?- `DLinear / Transformer-hybrid` 宸叉帴鍒?AGC 涓婄殑涓ょ被 MPC 姹傝В鍣?- `CEMMPC` 宸茶ˉ涓婂浐瀹氶殢鏈虹瀛愩€亀arm start銆乧andidate injection 鍜屾洿骞虫粦鐨?CEM 鏇存柊
- 闂幆 rollout 榛樿鍒囧埌鏇翠弗鏍肩殑 `surrogate` 妯″紡锛屼笉鍐嶉粯璁ょ敤鐪熷疄涓嬩竴琛岀姸鎬佹墦搴?- surrogate 鐘舵€佹洿鏂伴噷浼氶噸绠?`HumDef`锛屽苟鐢?persistence + action proxy 鏇存柊闈炵洰鏍囩姸鎬?- 鎺у埗缁撴灉鑷姩淇濆瓨鍒?`agc_mpc/results/control`
- 宸叉柊澧?`benchmark_hybrid_residual_forecaster.py`锛岀敤浜庡湪鍏钩璁粌棰勭畻涓嬪崟鐙瘎浼版贩鍚堟畫宸ā鍨?
褰撳墠鏈畬鎴愶細

- 瀹屾暣鐗╃悊绾?/ economic 绾?AGC 闂幆鐜
- 鏇翠弗鏍肩殑 actuator / VIP / resource-aware AGC 鎺у埗寤烘ā
- 璧勬簮鎴愭湰 / 缁忔祹鎸囨爣绾冲叆鎺у埗鐩爣


## 7. 褰撳墠榛樿瀹為獙璁剧疆

鏉ヨ嚜 [config.py](c:/repositories/strawberry/agc_mpc/config.py)锛?
- 榛樿闅旈棿锛? 涓叏閮ㄨ仈鍚堣缁?- `seq_len = 288`  
  鍚箟锛?4 灏忔椂鍘嗗彶绐楀彛
- `horizon = 24`  
  鍚箟锛? 灏忔椂棰勬祴绐楀彛
- 杩欐剰鍛崇潃鈥滃崟涓?forecast 绐楀彛鍥锯€濆ぉ鐒跺彧浼氭樉绀?24 涓湭鏉ユ锛涘鏋滄兂鐪嬫洿闀挎椂闂磋酱锛岄渶瑕佺湅 rolling forecast rollout 鍥撅紝鎴栫洿鎺ユ妸 `horizon` 鏀瑰ぇ鍚庨噸璁?- `batch_size = 256`
- `num_epochs = 12`
- `early_stop_patience = 4`
- `control_eval_steps = 96`
- `control_rollout_mode = surrogate`

褰撳墠鐩爣鍙橀噺锛?
- `Tair`
- `Rhair`
- `CO2air`
- `Tot_PAR`


## 8. 鏈€鏂板熀绾跨粨鏋?
鏈€鏂拌繍琛屾柟寮忥細

```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\main.py
```

鏈€鏂版暟鎹妯★細

- 6 涓殧闂磋仈鍚堣缁?- `train = 199488`
- `val = 40878`
- `test = 40878`

### 8.1 GRU baseline

- `Tair`: Full `R虏=0.9293`, MAE `0.886`; Final `R虏=0.9136`, MAE `1.026`
- `Rhair`: Full `R虏=0.8277`, MAE `3.996`; Final `R虏=0.7424`, MAE `5.067`
- `CO2air`: Full `R虏=0.7718`, MAE `55.797`; Final `R虏=0.7092`, MAE `64.391`
- `Tot_PAR`: Full `R虏=0.9688`, MAE `37.947`; Final `R虏=0.9660`, MAE `39.784`

缁撴灉鍥撅細

- [gru_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/gru_baseline_forecast_examples.png)
- [gru_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/gru_baseline_horizon_mae.png)

### 8.2 DLinear baseline

- `Tair`: Full `R虏=0.9639`, MAE `0.638`; Final `R虏=0.9526`, MAE `0.729`
- `Rhair`: Full `R虏=0.8607`, MAE `3.684`; Final `R虏=0.8184`, MAE `4.209`
- `CO2air`: Full `R虏=0.8205`, MAE `48.084`; Final `R虏=0.7928`, MAE `51.481`
- `Tot_PAR`: Full `R虏=0.9790`, MAE `30.483`; Final `R虏=0.9779`, MAE `31.295`

缁撴灉鍥撅細

- [dlinear_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_baseline_forecast_examples.png)
- [dlinear_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_baseline_horizon_mae.png)

### 8.3 SegRNN baseline

- `Tair`: Full `R虏=0.9228`, MAE `0.944`; Final `R虏=0.9076`, MAE `1.069`
- `Rhair`: Full `R虏=0.7512`, MAE `4.951`; Final `R虏=0.6662`, MAE `5.922`
- `CO2air`: Full `R虏=0.7856`, MAE `53.093`; Final `R虏=0.7176`, MAE `62.168`
- `Tot_PAR`: Full `R虏=0.9689`, MAE `38.705`; Final `R虏=0.9672`, MAE `40.208`

缁撴灉鍥撅細

- [segrnn_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/segrnn_baseline_forecast_examples.png)
- [segrnn_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/segrnn_baseline_horizon_mae.png)

### 8.4 绾?Transformer baseline

- `Tair`: Full `R虏=0.9483`, MAE `0.765`; Final `R虏=0.9413`, MAE `0.823`
- `Rhair`: Full `R虏=0.8038`, MAE `4.249`; Final `R虏=0.7454`, MAE `4.919`
- `CO2air`: Full `R虏=0.8509`, MAE `43.206`; Final `R虏=0.8242`, MAE `47.229`
- `Tot_PAR`: Full `R虏=0.9853`, MAE `26.484`; Final `R虏=0.9859`, MAE `24.964`

缁撴灉鍥撅細

- [transformer_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_baseline_forecast_examples.png)
- [transformer_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_baseline_horizon_mae.png)

### 8.5 Transformer-hybrid baseline

- `Tair`: Full `R虏=0.9544`, MAE `0.708`; Final `R虏=0.9480`, MAE `0.770`
- `Rhair`: Full `R虏=0.7539`, MAE `4.650`; Final `R虏=0.6927`, MAE `5.306`
- `CO2air`: Full `R虏=0.7870`, MAE `51.905`; Final `R虏=0.7434`, MAE `58.318`
- `Tot_PAR`: Full `R虏=0.9848`, MAE `28.237`; Final `R虏=0.9846`, MAE `28.509`

缁撴灉鍥撅細

- [transformer_hybrid_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_hybrid_baseline_forecast_examples.png)
- [transformer_hybrid_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_hybrid_baseline_horizon_mae.png)

褰撳墠绂荤嚎缁撹锛?
- `DLinear` 浠嶇劧鏄?`Tair / Rhair` 涓婃渶绋崇殑鏁翠綋 baseline
- 绾?`Transformer` 鍦ㄥ綋鍓嶈缃笅瀵?`CO2air / Tot_PAR` 鏈€寮猴紝涓旀暣浣撳己浜庡綋鍓?`Transformer-hybrid`
- `Transformer-hybrid` 浠嶄繚鐣欑粨鏋勪环鍊硷紝浣嗗綋鍓嶅疄鐜版病鏈夊湪鎵€鏈夌洰鏍囦笂瓒呰繃绾?Transformer
- `GRU` 褰撳墠涓嶅啀鏄暣浣撴渶浼橈紝浣嗕粛鐒舵槸閲嶈鐨勬椂搴?baseline
- `SegRNN` 褰撳墠鏈秴杩囧墠涓夎€?- 杩欑户缁敮鎸佷竴涓噸瑕佽鏂囪鐐癸細  
  **鏈€濂界殑绂荤嚎棰勬祴妯″瀷鍙兘鍥犵洰鏍囧彉閲忎笉鍚岃€屽垎鍖栵紝涓嶅瓨鍦ㄥ崟涓€缁濆鏈€浼樼粨鏋?*

### 8.6 鎺у埗渚?benchmark锛?026-03-23, stricter surrogate update锛?
杩愯鏂瑰紡锛?
```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\control_main.py --steps 48 --start-idx 0 --reference-mode trajectory
```

鍗忚璇存槑锛?
- 鎺у埗闅旈棿锛歚Reference`
- 鎺у埗鍣細`recorded` / `GradientMPC` / `CEMMPC`
- 棰勬祴鍣細`DLinear`銆佺函 `Transformer`銆乣Transformer-hybrid`
- 鍙傝€冪洰鏍囷細娴嬭瘯闆嗙湡瀹炴湭鏉?`y_future` trajectory
- 褰撳墠闂幆鍗忚浠嶄笉鏄畬鏁寸墿鐞嗕豢鐪熷櫒锛屼絾姣斾笂涓€鐗堟洿涓ユ牸锛?  - 澶╂皵銆佹椂闂村拰鍙傝€冭建杩圭户缁潵鑷?AGC 娴嬭瘯闆嗙湡瀹炲簭鍒?  - 琚帶鐩爣鐘舵€佺敱棰勬祴鍣ㄤ竴姝ユ粴鍔ㄤ骇鐢?  - 鍘嗗彶鐘舵€侀粯璁や笉鍐嶇洿鎺ユ嫹璐濈湡瀹炰笅涓€琛岋紝鑰屾槸浠庡綋鍓嶇姸鎬佸嚭鍙戯紝鐢?persistence + action proxy + predicted targets 鏇存柊
  - `HumDef` 鏍规嵁棰勬祴鐨?`Tair / Rhair` 閲嶆柊璁＄畻
  - `CEMMPC` 鐜板湪浣跨敤鍥哄畾闅忔満绉嶅瓙锛屽洜姝ゅ悓涓€鍛戒护閲嶈窇鏃?summary 鍝堝笇淇濇寔涓€鑷?
缁撴灉锛?
鏈璇存槑锛?
- 杩欓噷鍘熸潵鍐欎綔 `DPC` 鐨勬柟娉曪紝鐜板湪缁熶竴璁颁负 `GradientMPC`
- 瀹冧笉鏄嫭绔嬩簬 MPC 鐨勫彟涓€绫绘帶鍒惰寖寮忥紝鑰屾槸鈥滈€氳繃姊害鐩存帴姹傝В婊氬姩鏃跺煙浼樺寲闂鐨?MPC 姹傝В鍣ㄢ€?- 鍘熸潵鍐欎綔 `MPC(CEM)` 鐨勬柟娉曪紝鐜板湪缁熶竴璁颁负 `CEMMPC`
- 鍥犳褰撳墠鎺у埗瀵规瘮鏇村噯纭湴璇存槸锛歚GradientMPC vs CEMMPC`

#### DLinear as control surrogate

- `recorded`: `Tair=0.527`, `Rhair=4.533`, `CO2air=66.691`, `Tot_PAR=36.827`
- `GradientMPC`: `Tair=0.167`, `Rhair=0.458`, `CO2air=2.592`, `Tot_PAR=11.558`
- `CEMMPC`: `Tair=0.303`, `Rhair=1.237`, `CO2air=11.970`, `Tot_PAR=12.647`

#### Pure Transformer as control surrogate

- `recorded`: `Tair=1.482`, `Rhair=4.004`, `CO2air=29.998`, `Tot_PAR=20.712`
- `GradientMPC`: `Tair=0.251`, `Rhair=2.818`, `CO2air=15.884`, `Tot_PAR=18.789`
- `CEMMPC`: `Tair=0.423`, `Rhair=3.800`, `CO2air=21.208`, `Tot_PAR=24.774`

#### Transformer-hybrid as control surrogate

- `recorded`: `Tair=2.253`, `Rhair=3.134`, `CO2air=28.444`, `Tot_PAR=26.593`
- `GradientMPC`: `Tair=0.194`, `Rhair=1.861`, `CO2air=18.029`, `Tot_PAR=12.909`
- `CEMMPC`: `Tair=0.642`, `Rhair=4.026`, `CO2air=20.468`, `Tot_PAR=12.491`

褰撳墠鎺у埗缁撹锛?
- 鍦ㄦ洿涓ユ牸鐨?48-step surrogate rollout 涓婏紝`GradientMPC` 浠嶆櫘閬嶄紭浜?`CEMMPC`
- `DLinear + GradientMPC` 鏄綋鍓嶆渶寮虹殑涓ユ牸 surrogate 鎺у埗缁勫悎锛屽洓涓洰鏍囬兘鏄捐憲浼樹簬 recorded
- `CEMMPC` 鐜板湪宸茬粡鍙鐜帮紝鍚屼竴鍛戒护閲嶅杩愯鏃跺叾 summary 鍝堝笇淇濇寔涓€鑷达紝浣嗘€ц兘浠嶈惤鍚庝簬 `GradientMPC`
- surrogate 鍗忚涓€鏃︽敹绱э紝recorded control 鍜屽悇 predictor 鐨勮宸兘浼氭槑鏄惧彉澶э紝杩欒鏄庝笂涓€鐗?semi-grounded rollout 纭疄鍋忎箰瑙?- 杩欒繘涓€姝ユ彁绀猴細**鏈€寮虹绾块娴嬪櫒涓嶄竴瀹氳嚜鍔ㄥ彉鎴愭渶寮洪棴鐜帶鍒?surrogate**


## 9. 褰撳墠鑰楁椂缁忛獙鍊?
鍦ㄥ綋鍓嶆満鍣ㄥ拰褰撳墠閰嶇疆涓嬶細

- 鍗曢殧闂?GRU baseline锛氱害 `20 绉抈
- 6 闅旈棿鑱斿悎 `GRU + DLinear + SegRNN`锛氱害 `136 绉抈
- 6 闅旈棿鑱斿悎 `GRU + DLinear + SegRNN + Transformer-hybrid`锛氱害 `541 绉抈

绮楃暐浼拌锛?
- 杞婚噺 baseline锛歚2 鍒嗛挓鍐卄
- 涓瓑瑙勬ā GRU / SegRNN锛歚2~5 鍒嗛挓`
- Transformer / hybrid锛歚6~10 鍒嗛挓`


## 10. 褰撳墠璁烘枃瀹氫綅

褰撳墠鏈€浼樺畾浣嶄笉鏄細

- 鈥滄敼杩?Transformer 鍋氭俯瀹ら娴嬧€?
鏇村悎鐞嗙殑瀹氫綅鏄細

- 鈥滈潰鍚戞帶鍒剁殑娓╁澶氭棰勬祴鈥?- 鈥滃埄鐢ㄦ湭鏉ュぉ姘斾笌鏈潵鎺у埗淇℃伅鐨勯棴鐜娴嬫帶鍒舵鏋垛€?- 鈥滈娴嬫ā鍨嬩笌鎺у埗鎬ц兘涔嬮棿鍏崇郴鐨勭郴缁?benchmark鈥?

## 11. 褰撳墠鍒涙柊鐐瑰垽鏂?
### 鍙互鎴愮珛鐨勫垱鏂扮偣

- 闈㈠悜鎺у埗鐨勫姝ラ娴嬶紝鑰屼笉鏄函绂荤嚎鎷熷悎
- 鏄惧紡鍒╃敤鏈潵澶╂皵鍜屾湭鏉ユ帶鍒?- 涓ユ牸闂幆璇勪及
- 棰勬祴妯″瀷涓庢帶鍒剁粨鏋滀箣闂村樊寮傜殑绯荤粺鍒嗘瀽
- 澶氬彉閲忚€﹀悎寤烘ā

### 涓嶈兘鍗曠嫭浣滀负寮哄垱鏂扮偣鐨勫唴瀹?
- 鈥滅敤浜?Transformer鈥?- 鈥滃鍙傛暟鑰﹀悎鈥?- 鈥滅敤浜?SAC baseline鈥?- 鈥滃仛浜?MPC鈥?
杩欎簺鍙兘浣滀负鑳屾櫙鎴栫粍鎴愰儴鍒嗭紝涓嶈兘鍗曠嫭鎾戣捣璁烘枃涓诲垱鏂般€?

## 12. 褰撳墠浼樺厛绾?
### 绗竴浼樺厛绾?
鍏堢ǔ浣忔帶鍒?benchmark锛?
- 宸插畬鎴愶細`CEMMPC` 鐨勫彲澶嶇幇鎬у拰鍩虹绋冲畾鎬?- 姝ｅ湪鍋氾細楠岃瘉 `DLinear / Transformer / Transformer-hybrid` 鍦ㄦ洿闀?rollout 涓嬬殑闂幆鎺掑悕
- 涓嬩竴姝ワ細閫愭鎶?`sp -> actuator feedback -> climate` 鐨?surrogate 鏇存柊鍋氬疄

绗簩灞傜户缁ˉ寮洪娴?benchmark锛?
- 宸插惎鍔?`hybrid residual model`
- 涓嬩竴姝ユ槸缁?`hybrid residual model` 璺戞寮忛绠楋紝骞朵笌 `DLinear / Transformer / current hybrid-transformer` 鍋氱粺涓€鍙ｅ緞瀵规瘮

### 绗簩浼樺厛绾?
鎶?AGC 涓荤嚎鎺ュ埌鎺у埗灞傦細

- 浠庡綋鍓?surrogate closed-loop 缁х画鎺ㄨ繘鍒版洿涓ユ牸鐨?AGC 闂幆鐜
- 鍐嶇湅 SAC on AGC

### 绗笁浼樺厛绾?
鎶婅祫婧愭寚鏍囩撼鍏ワ細

- `Heat_cons`
- `ElecHigh`
- `ElecLow`
- `CO2_cons`
- `Irr`

鍚?economic MPC 寤朵几銆?

## 13. 褰撳墠宸ヤ綔瑙勫垯

1. 鏂板紑鍙戜紭鍏堟斁鍦?[agc_mpc](c:/repositories/strawberry/agc_mpc)銆?2. 闄ら潪鏈夋槑纭渶瑕侊紝涓嶈缁х画鎶婁富宸ヤ綔娴佸爢鍥?`diffmpc`銆?3. 浠ｇ爜榛樿杩愯鐜鏄?`strawberry_env`銆?4. 姣忔鍋氬畬鍏抽敭浠ｇ爜鏀瑰姩銆佸疄楠岀粨鏋滄洿鏂版垨璺嚎鍙樺寲鍚庯紝閮借鏇存柊鏈枃浠躲€?5. 褰撳墠鎺у埗鏈绾﹀畾锛?   - `GradientMPC` = 閫氳繃姊害鐩存帴姹傝В婊氬姩鏃跺煙浼樺寲闂鐨?MPC 姹傝В鍣?   - `CEMMPC` = 閫氳繃 CEM 閲囨牱鎼滅储姹傝В鍚屼竴 MPC 鐩爣鐨?MPC 姹傝В鍣?   - 涓嶅啀鎶?`DPC` 鍜?`MPC` 璁版垚涓や釜骞崇骇鑼冨紡锛屼互鍏嶆湳璇贩娣?6. 浠讳綍鏂版ā鍨嬮兘瑕佸悓鏃跺洖绛斿洓涓棶棰橈細
   - 绂荤嚎棰勬祴鏄惁鎻愬崌
   - 闂幆鎺у埗鏄惁鎻愬崌
   - 瀵?forecast error 鏄惁绋冲仴
   - 鏄惁鑳借В閲婁负闈㈠悜鎺у埗鐨勮璁?7. Git 鎻愪氦榛樿閲囩敤鈥滃皬姝ュ垎娈垫彁浜も€濓紝涓嶈鎶婄粨鏋滅洰褰曢噸鏋勩€佹ā鍨嬫柊澧炪€佹帶鍒跺疄楠岀粨鏋溿€佹枃妗ｆ洿鏂颁竴娆℃€ф贩鎴愪竴涓ぇ鎻愪氦銆?8. 褰撳墠浠撳簱鍦ㄦ湰鏈轰笂鏇惧嚭鐜?`.git` ACL / `index.lock` 鍐欏叆鍙楅檺闂锛涘鏋?`git add` / `git commit` 鎶?`Unable to create .git/index.lock: Permission denied`锛?   - 涓嶈鍙嶅閲嶈瘯寰堝娆?   - 鍏堟鏌?`.git` 鐨?ACL
   - 蹇呰鏃朵竴娆℃€ч€掑綊绉婚櫎 `.git` 涓嬮拡瀵瑰綋鍓嶇敤鎴风殑 `DENY` ACL 鍚庡啀缁х画鎻愪氦
9. 鎺ㄨ崘鐨勬彁浜ゆ媶鍒嗛『搴忥細
   - 鍏堟彁缁撴灉鐩綍缁撴瀯 / plotting / 鍩虹璁炬柦
   - 鍐嶆彁鏂版ā鍨嬩笌 forecasting 缁撴灉
   - 鏈€鍚庢彁 control benchmark銆佺粨鏋滃浘 / summary 鍜?`CONTEXT.md`
10. 濡傛灉鍚庣画 push 鍥?pack 杩囧ぇ鎴栦簩杩涘埗缁撴灉杩囧澶辫触锛屼紭鍏堣€冭檻缁х画鎷嗘彁浜わ紝蹇呰鏃舵妸鈥滀唬鐮佸彉鏇粹€濆拰鈥滃疄楠屼骇鐗┾€濆垎寮€澶勭悊锛岃€屼笉鏄棤闄愰噸璇?push銆?11. 褰撳墠鐜涓嬶紝`Remove-Item` 涓€绫诲垹闄ゅ姩浣滀篃鍙兘琚矙绠辨嫤浣忓苟鎶?`Access is denied`锛屽嵆浣挎枃浠?ACL 鐪嬭捣鏉ユ甯革紱濡傛灉闇€瑕佹竻鐞?legacy 缁撴灉鏂囦欢锛?   - 鍏堝尯鍒嗘槸娌欑/鎻愭潈闄愬埗杩樻槸鏂囦欢鑷韩 ACL 闂锛屼笉瑕侀粯璁ゆ槸鏂囦欢鎹熷潖
   - 浼樺厛鐢ㄢ€滅簿纭繃婊?+ 鎻愭潈鍒犻櫎鈥濓紝涓嶈鐢ㄤ細璇激鏂版枃浠剁殑瀹芥硾閫氶厤
   - 渚嬪娓呯悊鏃ф帶鍒剁粨鏋滄椂锛屽彧鍒犻櫎鏃у懡鍚嶇殑 `_dpc_` 鍜屾棫 `_mpc_` 鏂囦欢锛屼笉瑕佸尮閰嶅埌 `gradient_mpc` / `cem_mpc`


## 14. 涓嬫瀵硅瘽寤鸿璧锋墜鍐呭

寤鸿鍏堣鏄庯細

- 褰撳墠涓婚」鐩洰褰曪細`agc_mpc`
- 褰撳墠涓绘暟鎹泦锛歚AutonomousGreenhouseChallenge_edition2`
- 褰撳墠宸插畬鎴愶細鏁版嵁绠＄嚎 + GRU baseline + DLinear baseline
- 褰撳墠宸插畬鎴愶細鏁版嵁绠＄嚎 + GRU baseline + DLinear baseline + SegRNN baseline + Transformer baseline + Transformer-hybrid baseline + hybrid residual 鍘熷瀷 + 鑷姩缁撴灉鍥?- 褰撳墠宸插畬鎴愶細`DLinear / Transformer / Transformer-hybrid` 宸叉帴鍏?AGC 涓婄殑 `GradientMPC / CEMMPC` 鍒濈増 surrogate closed-loop benchmark
- 褰撳墠宸插畬鎴愶細forecast 渚ф柊澧?rolling multi-step rollout 鍥撅紱control 渚ч粯璁ゅ垏鍒版洿涓ユ牸鐨?`surrogate` rollout锛屽苟楠岃瘉浜?`CEMMPC` 鐨勫彲澶嶇幇鎬?- 褰撳墠涓嬩竴姝ワ細鎺у埗渚х户缁妸 surrogate 浠?`state persistence + action proxy` 鎺ㄥ埌鏇存帴杩?`sp -> vip -> actuator -> climate` 鐨勫眰绾у缓妯★紱棰勬祴渚ф妸 `hybrid residual model` 璺戞垚姝ｅ紡棰勭畻骞跺仛缁熶竴瀵规瘮
## 15. Strawberry vs AGC 瀵规瘮鍥?
- 宸叉柊澧炲甯堝睍绀虹敤鑴氭湰锛歔compare_dataset_switch.py](c:/repositories/strawberry/agc_mpc/compare_dataset_switch.py)
- 杩愯鏂瑰紡锛?  ```bash
  conda activate strawberry_env
  python c:\repositories\strawberry\agc_mpc\compare_dataset_switch.py
  ```
- 杈撳嚭鏂囦欢锛?  - [strawberry_vs_agc_dataset_switch.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/strawberry_vs_agc_dataset_switch.png)
  - [strawberry_vs_agc_dataset_switch_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/strawberry_vs_agc_dataset_switch_summary.json)
- 鍥剧殑姣旇緝鍙ｅ緞锛?  - 鍙瘮杈冨叡鍚屽彉閲?`Temperature / Humidity / CO2`
  - 鍙瘮杈?final-step 鎸囨爣
  - 涓よ竟閮芥寜鈥? 灏忔椂棰勬祴浠诲姟鈥濆榻愶細鏃?Strawberry = `120 x 1 min`锛孉GC = `24 x 5 min`
- 褰撳墠缁撹锛?  - 鏃?Strawberry Transformer-hybrid 鐨?final MAE 涓?`3.36 / 6.78 / 105.88`
  - AGC `DLinear` 鐨?final MAE 涓?`0.76 / 4.46 / 54.73`
  - AGC `Transformer` 鐨?final MAE 涓?`0.82 / 4.92 / 47.23`
  - AGC `Transformer-hybrid` 鐨?final MAE 涓?`0.77 / 5.31 / 58.32`
  - 鏃?Strawberry 鍦?`CO2` 涓?final `R2` 鍙湁 `0.073`锛汚GC 涓変釜妯″瀷瀵瑰簲涓?`0.776 / 0.824 / 0.743`
- 瀵瑰甯堢殑鎺ㄨ崘琛ㄨ堪锛?  - 杩欏紶鍥句笉璇佹槑 鈥淎GC 宸茬粡鍋氬埌鐞嗘兂鏋侀檺鈥?  - 瀹冭瘉鏄庣殑鏄細鍦ㄥ綋鍓?baseline-first 瀹炵幇涓嬶紝AGC 宸茬粡鑳芥彁渚涙洿绋冲畾銆佹洿鍙帶銆佸闂幆鏇村弸濂界殑棰勬祴鍩哄骇
- 鍥犳鍒囨崲鏁版嵁闆嗙殑涓昏鐞嗙敱搴旇〃杩颁负鈥滀换鍔″尮閰嶅害鏇撮珮 + 缁撴灉鏇寸ǔ + 鑳借嚜鐒舵墿灞曞埌闂幆鎺у埗鈥濓紝鑰屼笉鍙槸鈥滄棫鏁版嵁闆嗗垎鏁板樊鈥?- 宸叉柊澧炰唬琛ㄦ€ч娴嬬獥瀵规瘮鍥撅細[strawberry_vs_agc_forecast_windows.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/strawberry_vs_agc_forecast_windows.png)
- 璇ュ浘鍙睍绀?`Strawberry / old Transformer-hybrid`銆乣AGC / Transformer`銆乣AGC / Transformer-hybrid`
- 璇ュ浘浣跨敤涓よ竟娴嬭瘯闆嗗悇鑷殑 midpoint sample锛屼笉鍋氭牱鏈榻愶紝涓嶇敤浜庝弗鏍肩粺璁℃瘮杈冿紝鍙敤浜庣粰瀵煎笀鍋氣€滈娴嬭建杩瑰舰鎬佲€濈殑鐩磋璇存槑
- forecasting 鍥惧凡鍗囩骇涓衡€滃浘鍐呯洿鎺ユ樉绀烘寚鏍団€濓細
- `forecast_examples / rollout / horizon_mae` 鐜板湪閮戒細鐩存帴鍏宠仈褰撳墠妯″瀷鐨?`Full R2 / Full MAE / Final R2 / Final MAE`
  - `horizon_mae` 鍥句細鍦ㄥ浘涓嬫柟姹囨€诲叏閮ㄧ洰鏍囩殑鎸囨爣
- control 鍥惧凡鍗囩骇涓衡€滅姸鎬?+ 鎸囨爣 + 鍔ㄤ綔鈥濊仈鍚堝睍绀猴細
  - 鍓嶅洓琛屼粛鏄?`Tair / Rhair / CO2air / Tot_PAR`
  - 绗簲琛屾樉绀?`objective / |u-u_log| / action_tv`
  - 绗叚琛屾樉绀烘墍鏈夋帶鍒堕噺鐨勫綊涓€鍖栧姩浣滆建杩癸紝瀹炵嚎鏄?executed锛岃櫄绾挎槸 logged baseline
- 宸叉柊澧炴枃鐚鐓ф枃妗ｏ細[LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md)
- 鏂囩尞瀵圭収鏂囨。鐨勫畾浣嶏細
  - 涓嶅仛浼?leaderboard
  - 鎸変换鍔°€佽緭鍏ャ€佽緭鍑恒€乭orizon銆佹ā鍨嬨€佹帶鍒惰瀹氥€佺粨鏋滃拰鍙瘮鎬у垎寮€鍐?  - 褰撳墠缁撹鏄細AGC 缁撴灉杩樹笉鏄?final-paper quality锛屼絾宸插浜庡彲杈╂姢鐨?literature band 鍐咃紱鐪熸鐭澘鍦?`Rhair`銆乽ncertainty銆乪conomic objective 鍜屾洿瀹屾暣闂幆
- 宸叉柊澧炶〃鏍煎紡杩戞湡璁烘枃缁艰堪鏂囨。锛歔RECENT_PAPERS_SURVEY.md](c:/repositories/strawberry/agc_mpc/RECENT_PAPERS_SURVEY.md)
- 璇ユ枃妗ｆ寜鈥滆鏂?/ 浠诲姟 / 涓绘ā鍨?/ 瀵规瘮-baseline / 鍚彂 / 閾炬帴鈥濈粍缁囷紝鍒嗕负锛?  - 娓╁棰勬祴璁烘枃
  - 娓╁鎺у埗璁烘枃
  - 閫氱敤鏃跺簭妯″瀷鍙傝€?- 鐢ㄩ€旓細
  - 蹇€熷洖绛斺€滄渶杩戠浉浼艰鏂囬兘鐢ㄤ簡浠€涔堟ā鍨嬨€乥aseline 鍜屽姣斿璞℃槸浠€涔堚€?  - 涓哄悗缁柊妯″瀷璺嚎鎻愪緵鏂囩尞閿氱偣锛岄伩鍏嶅弽澶嶆墜宸ユ暣鐞?- 宸插湪璇ユ枃妗ｄ腑琛ュ厖 `Mao et al., 2024` 鐨勯噸鐐硅瑙ｅ皬鑺傦紝涓撻棬鍥炵瓟锛?  - 涓轰粈涔堣鏂?`PSO-BiGRU-Attention-LightGBM` 鐨?`R2` 寰堥珮
  - 瀹冨拰褰撳墠 `AGC` 鏁版嵁闆嗗埌搴曟湁澶氱浉浼?  - 瀹冩槸鍚﹀彲浠ヨ涓ユ牸澶嶇幇锛屽摢浜涢儴鍒嗗彧鑳藉仛鏂规硶绾у鐜?- 璇ユ枃妗ｇ幇宸叉寜褰撳墠涓荤嚎琛ュ己骞堕噸鍐欎负骞插噣 UTF-8 鐗堟湰锛屾柊澧炴垨寮哄寲浜嗭細
  - `Zeng et al., 2022 / DLinear`
  - `PatchTST`
  - `iTransformer`
  - `TimeMixer`
  - `SAMformer`
  - `ETSformer`
  - `FreTS`
  - `OneNet`
- 褰撳墠鏇存槑纭殑鏂囩尞缁撹鏄細
  - 涓嶅€煎緱缁х画鍫?plain Transformer
  - 鏇村悎鐞嗙殑鏄?`DLinear main path + stronger residual branch`
  - 褰撳墠鏈€鍊煎緱浼樺厛璺戞竻妤氱殑涓夋潯 residual 璺嚎鏄?`Transformer-hybrid / iTransformer / PatchTST residual`
- `README.md` 宸茶ˉ鍏呮暟鎹泦鑳屾櫙涓庤缁冭瀹氳鏄庯細
  - 鏄庣‘ `AGC` 鏇村噯纭槸 multi-compartment benchmark锛岃€屼笉鏄?fully independent multi-greenhouse dataset
  - 琛ュ厖褰撳墠 `x_past / w_future / u_future / y_future` 鐨勬帶鍒跺鍚戞帴鍙ｈ鏄?  - 琛ュ厖 single-compartment training 涓?multi-compartment joint training 鐨勫彇鑸嶏紝褰撳墠榛樿浠嶄互 joint training 涓轰富
- 宸叉柊澧炶缁冪瓥鐣ュ鐓ц剼鏈細[compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py)
- 璇ヨ剼鏈敮鎸佸洿缁曚竴涓洰鏍囬殧闂存瘮杈冧笁绉?regime锛?  - `single`: 鍙湪鐩爣闅旈棿涓婅缁冨苟鍦ㄨ闅旈棿娴嬭瘯
  - `joint_all`: 鍦ㄥ叏閮ㄩ殧闂翠笂璁粌锛屼絾鍙湪鐩爣闅旈棿娴嬭瘯
  - `leave_one_out`: 鍦ㄩ櫎鐩爣闅旈棿澶栫殑鍏朵綑闅旈棿涓婅缁冿紝鍐嶅湪鐩爣闅旈棿娴嬭瘯
- 鏁版嵁绠＄嚎宸叉柊澧炶嚜瀹氫箟 bundle 缁勮鑳藉姏锛屽彲鎸?train/eval compartments 鑷敱鎷兼帴骞朵粎鐢ㄨ缁冮泦鎷熷悎 scaler
- 缁撴灉缁熶竴钀藉埌锛歚agc_mpc/results/forecasting/analysis`
- 宸插仛 1-epoch smoke test锛堢洰鏍囬殧闂?`Reference`锛屾ā鍨?`DLinear`锛夛細
  - `single`锛歚Tair/Rhair/CO2air/Tot_PAR` Final MAE = `0.772 / 4.815 / 93.219 / 53.889`
  - `joint_all`锛歚0.776 / 3.798 / 53.866 / 32.658`
  - `leave_one_out`锛歚0.671 / 5.469 / 56.336 / 38.663`
- 鍒濇淇″彿锛?  - joint training 瀵?`Rhair / CO2air / Tot_PAR` 鏄庢樉鏇存湁甯姪
  - leave-one-out 鍦?`Reference` 鐨?`Tair` 涓婂緢寮猴紝浣嗗婀垮害鍜?CO2 涓嶅崰浼?  - 鍗曢殧闂磋缁冨苟涓嶅ぉ鐒舵洿濂斤紝鑷冲皯鍦ㄥ綋鍓?`Reference + DLinear` 鐨?smoke test 涓婁笉鏄?- 宸叉柊澧?`diffmpc` 椋庢牸 Transformer 杩佺Щ鍩哄噯鑴氭湰锛歔benchmark_diffmpc_style_transformer.py](c:/repositories/strawberry/agc_mpc/benchmark_diffmpc_style_transformer.py)
- 璇ヨ剼鏈殑鐩殑涓嶆槸杩藉綋鍓?`agc_mpc` 鏈€寮哄垎鏁帮紝鑰屾槸鎺у埗鍙橀噺鍦板洖绛旓細
  - 鍦ㄥ敖閲忎繚鐣欐棫 `diffmpc` Transformer-hybrid 鏋舵瀯涓庤缁冮绠楁椂锛宍AGC` 鏄惁姣旀棫 Strawberry 鏇撮€傚悎浣滀负 Transformer 鐨勬暟鎹熀搴?- 鍥哄畾鍗忚锛?  - targets = `Tair / Rhair / CO2air`
  - `seq_len = 48`锛堝搴旀棫椤圭洰 `240 min` 鍘嗗彶锛?  - `horizon = 24`锛堝搴旀棫椤圭洰 `120 min` 棰勬祴绐楋級
  - `d_model = 64`, `nhead = 4`, `num_layers = 4`, `ff_dim = 128`, `dropout = 0.1`
  - `batch_size = 256`, `num_epochs = 200`, `lr = 1e-4`, `lambda_trend = 0.3`, `patience = 15`
- 璁捐鍘熷垯锛?  - 榛樿鍙惤 summary JSON锛屼笉鑷姩鐢熸垚澶у浘
  - 鍏堟妸鈥滄ā鍨嬬粨鏋?璁粌棰勭畻/鏃堕棿鍙ｅ緞鈥濆榻愶紝鍐嶈皥鏁版嵁闆嗘槸鍚︽洿閫傚悎 Transformer
- 宸插仛 1-epoch smoke test锛坄single + Reference`锛夊苟鎴愬姛钀界洏锛?  - [diffmpc_style_transformer_single_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_single_reference_summary.json)
  - 褰撳墠浠呯敤浜庨獙璇佸叆鍙ｄ笌鍗忚锛屼笉鐢ㄤ簬姝ｅ紡缁撹
- 璇ュ熀鍑嗙幇宸插畬鎴?`Reference` 涓婄殑姝ｅ紡涓夌粍杩愯锛?  - [diffmpc_style_transformer_single_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_single_reference_summary.json)
  - [diffmpc_style_transformer_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_joint_all_reference_summary.json)
  - [diffmpc_style_transformer_leave_one_out_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_leave_one_out_reference_summary.json)
- `diffmpc` 椋庢牸 Transformer 鍦?AGC / `Reference` 涓婄殑鏈€缁堟寚鏍囷細
  - `single`
    - `Tair`: Final `R2=0.5198`, MAE `2.023`
    - `Rhair`: Final `R2=0.6850`, MAE `6.874`
    - `CO2air`: Final `R2=0.3543`, MAE `75.857`
  - `joint_all`
    - `Tair`: Final `R2=0.8007`, MAE `1.358`
    - `Rhair`: Final `R2=0.6470`, MAE `7.891`
    - `CO2air`: Final `R2=0.3899`, MAE `72.867`
  - `leave_one_out`
    - `Tair`: Final `R2=0.8859`, MAE `0.926`
    - `Rhair`: Final `R2=0.5763`, MAE `8.169`
    - `CO2air`: Final `R2=0.3140`, MAE `78.422`
- 褰撳墠璇绘硶锛?  - 鏃?`diffmpc` 椋庢牸缁撴瀯杩佸埌 AGC 鍚庯紝`Tair / CO2air` 鏄庢樉濂戒簬鏃?Strawberry 涓婄殑鏃?Transformer-hybrid 缁撴灉锛岃鏄庢暟鎹泦鍒囨崲纭疄甯姪浜嗚繖绫?conditional Transformer
  - 浣?`Rhair` 娌℃湁鍚屾鍙樻垚寮洪」锛岃鏄庘€滄暟鎹泦鏇撮€傚悎 Transformer鈥濅笉绛変簬鈥滄棫缁撴瀯鏃犻渶鏀归€犲氨浼氬叏闈㈠彉寮衡€?  - 涓夌 AGC 璁粌 regime 娌℃湁鍗曚竴缁濆鏈€浼橈細
    - `single` 鍦?`Rhair` 涓婃渶濂?    - `joint_all` 鍦?`CO2air` 涓婃渶濂?    - `leave_one_out` 鍦?`Tair` 涓婃渶濂?  - 鍥犳瀵瑰甯堟洿绋崇殑琛ㄨ堪搴旀槸锛?    - AGC 缁欐棫 Transformer 椋庢牸鎻愪緵浜嗘洿鍚堢悊鐨勬暟鎹帴鍙ｅ拰鏇撮珮鐨勪笂闄愮┖闂?    - 浣嗙湡姝ｆ妸璇ユ灦鏋勫仛寮猴紝浠嶇劧闇€瑕佽繘涓€姝ラ潰鍚?AGC/鎺у埗浠诲姟鏀归€狅紝鑰屼笉鏄洿鎺ョ収鎼棫缁撴瀯
- 宸叉柊澧炵洿瑙傚姣斿浘锛歔diffmpc_style_transformer_dataset_suitability.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/diffmpc_style_transformer_dataset_suitability.png)
- 璇ュ浘鍙瘮杈冿細
  - `Strawberry / old Transformer-hybrid`
  - `AGC / diffmpc-style / single`
  - `AGC / diffmpc-style / joint_all`
  - `AGC / diffmpc-style / leave_one_out`
- 璇ュ浘鐨勫畾浣嶏細
  - 鐢ㄤ簬鐩磋灞曠ず鈥滃敖閲忕浉浼肩殑 Transformer 椋庢牸涓庤缁冮绠椻€濅笅锛屾崲鍒?AGC 鍚?`Temperature / Humidity / CO2` 鐨?final MAE 涓?final R2 濡備綍鍙樺寲
  - 涓嶆贩鍏ュ綋鍓?`agc_mpc` 鐨?`DLinear / Transformer / Transformer-hybrid` 鏂?baseline锛岄伩鍏嶈璇佸彛寰勬紓绉?- 宸叉柊澧炴洿閫傚悎姹囨姤鐨勪袱寮犱腑鏂囧浘锛?  - [diffmpc_style_transformer_best_vs_old_line_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/diffmpc_style_transformer_best_vs_old_line_cn.png)
  - [diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png)
- 涓ゅ浘瀹氫綅锛?  - `best_vs_old_line_cn`锛氬彧鐪?`old Strawberry` vs `AGC joint_all`锛岀敤涓枃鎶樼嚎鍥惧睍绀?`Temperature / Humidity / CO2` 鐨?final MAE 涓?final R2
  - `old_vs_agc_joint_all_windows_cn`锛氬苟鎺掑睍绀烘棫 Strawberry 涓?AGC joint_all 鐨勪唬琛ㄦ€ч娴嬬獥锛岃瀵煎笀鐩存帴鐪嬭建杩硅创鍚堜笌鍋忕Щ鏂瑰紡
- 宸叉柊澧炩€滄棫鏁版嵁闆嗘棫 hybrid-transformer vs 鏂版暟鎹泦鏂?hybrid-transformer鈥濈殑鍏钩棰勭畻瀵圭収涓荤嚎锛?  - 鏃т晶锛歚diffmpc` 鍘熷 `TransformerHybridModel`
  - 鏂颁晶锛歚agc_mpc` 褰撳墠 `ConditionalTransformerHybridForecaster`
  - 鍏卞悓鍙ｅ緞锛氬彧鐪?`Tair / Rhair / CO2air`锛岀粺涓€鎸?`2h` 棰勬祴浠诲姟璁ㄨ
  - 鏃т晶淇濈暀鏃ч」鐩柟娉曚笌鏋舵瀯锛涙柊渚т繚鐣?AGC 褰撳墠 `x_past / w_future / u_future -> y_future` 鐨?control-oriented 鎺ュ彛
- 宸叉柊澧炶剼鏈細[benchmark_current_hybrid_transformer.py](c:/repositories/strawberry/agc_mpc/benchmark_current_hybrid_transformer.py)
  - 鐩殑锛氱粰 AGC 褰撳墠 hybrid-transformer 涓€涓瘮 12 epoch baseline 鏇村叕骞崇殑璁粌棰勭畻锛屽啀涓庢棫 Strawberry 鐨?old hybrid-transformer 鍋氬姣?  - 褰撳墠姝ｅ紡璺戦€氱殑閰嶇疆涓猴細`joint_all + Reference`
  - 璁粌棰勭畻锛歚batch_size=256`, `num_epochs=200`, `lr=1e-4`, `lambda_trend=0.3`, `patience=15`
  - 妯″瀷鍙傛暟锛歚hidden_dim=96`, `nhead=4`, `num_layers=2`, `ff_dim=192`, `dropout=0.1`
- 褰撳墠姝ｅ紡缁撴灉鏂囦欢锛?  - [current_hybrid_transformer_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/current_hybrid_transformer_joint_all_reference_summary.json)
  - [current_hybrid_transformer_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/current_hybrid_transformer_joint_all_reference.pt)
- `AGC + current hybrid-transformer + joint_all + Reference` 姝ｅ紡缁撴灉锛?  - `Tair`: Full `R2=0.9344`, MAE `0.630`; Final `R2=0.9318`, MAE `0.651`
  - `Rhair`: Full `R2=0.8951`, MAE `3.698`; Final `R2=0.8553`, MAE `4.403`
  - `CO2air`: Full `R2=0.8184`, MAE `41.201`; Final `R2=0.7860`, MAE `44.567`
- 涓?`AGC + diffmpc-style hybrid-transformer + joint_all + Reference` 鐨勭洿鎺ュ姣旓細
  - `Tair`: Final MAE `1.358 -> 0.651`, Final `R2 0.8007 -> 0.9318`
  - `Rhair`: Final MAE `7.891 -> 4.403`, Final `R2 0.6470 -> 0.8553`
  - `CO2air`: Final MAE `72.867 -> 44.567`, Final `R2 0.3899 -> 0.7860`
- 褰撳墠鏇寸ǔ鐨勮〃杩板簲鏀逛负锛?  - 涓嶆槸鈥淎GC 鑷姩璁╂棫 Transformer 鍙樺己鈥?  - 鑰屾槸鈥淎GC 鏇撮€傚悎褰撳墠杩欏闈㈠悜鎺у埗鐨?hybrid-transformer 鎺ュ彛涓庤缁冭寖寮忊€?  - 鏃?Strawberry + old hybrid-transformer 涓?AGC + current hybrid-transformer 鐨勫姣旓紝鎵嶆洿鑳芥敮鎸佲€樻崲鏁版嵁闆?+ 鎹㈡柟娉曟槸鍚堢悊涓荤嚎鈥欒繖涓€缁撹
- 宸叉柊澧?`AGC + current hybrid-transformer + joint_all + Reference + horizon=120` 姝ｅ紡瀹為獙锛?  - [current_hybrid_transformer_h120_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/current_hybrid_transformer_h120_joint_all_reference_summary.json)
  - [current_hybrid_transformer_h120_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/current_hybrid_transformer_h120_joint_all_reference.pt)
  - 娉ㄦ剰锛氳繖閲岀殑 `120-step` 鎸?`120 x 5min = 600 min`锛屼笉鍐嶇瓑浠蜂簬鏃?Strawberry 鐨?`120 x 1min = 120 min`
- `AGC current hybrid-transformer` 鍦?`horizon=120` 涓嬬殑姝ｅ紡缁撴灉锛?  - `Tair`: Full `R2=0.9204`, MAE `0.764`; Final `R2=0.9153`, MAE `0.820`
  - `Rhair`: Full `R2=0.7302`, MAE `6.705`; Final `R2=0.7149`, MAE `6.875`
  - `CO2air`: Full `R2=0.5754`, MAE `63.666`; Final `R2=0.5573`, MAE `65.198`
- 涓庡綋鍓?`horizon=24` 瀵规瘮鐨勮娉曪細
  - `Tair`: Final MAE `0.651 -> 0.820`
  - `Rhair`: Final MAE `4.403 -> 6.875`
  - `CO2air`: Final MAE `44.567 -> 65.198`
  - 璇存槑锛氭妸 AGC 浠诲姟浠?`2h` 鎷夊埌 `10h` 鍚庯紝鎬ц兘鏄庢樉涓嬮檷锛屼絾 `Tair` 浠嶄繚鎸佽緝寮猴紱`Rhair / CO2air` 鏇村鏄撻殢 horizon 鎷夐暱鑰岄€€鍖?- 宸叉柊澧炰袱寮犱腑鏂?horizon 瀵规瘮鍥撅細
  - [current_hybrid_transformer_h24_vs_h120_metrics_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h24_vs_h120_metrics_cn.png)
  - [current_hybrid_transformer_h24_vs_h120_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h24_vs_h120_windows_cn.png)
  - 鐢ㄩ€旓細鎶?`24-step (120 min)` 鍜?`120-step (600 min)` 鏀惧湪鍚屼竴椤典笂锛岀湅鎸囨爣鍜岃建杩瑰浣曢殢 horizon 鎷夐暱鑰岄€€鍖?- 宸叉柊澧炴洿绗﹀悎褰撳墠涓荤嚎鐨勪袱寮犱腑鏂囨眹鎶ュ浘锛?  - [current_hybrid_transformer_best_vs_old_line_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_best_vs_old_line_cn.png)
  - [current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png)
- 宸叉柊澧炩€滃垎閽熷榻愬睍绀虹増鈥濈獥鍙ｅ浘锛?  - [current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png)
  - 鐢ㄩ€旓細璁╁乏渚?`120 x 1min` 鍜屽彸渚?`24 x 5min` 鍦ㄨ瑙変笂閮藉睍寮€鍒?`120 min` 鏃堕棿杞达紝渚夸簬瀵煎笀鑲夌溂姣旇緝
  - 閲嶈璇存槑锛氬彸渚у彧鏄妸 `24 x 5min` 鐨勭湡瀹?棰勬祴杞ㄨ抗鎻掑€煎埌 `120` 涓垎閽熺偣鍋氭樉绀猴紝涓嶄唬琛ㄦā鍨嬬湡鐨勫仛浜?`120` 姝?AGC 棰勬祴
- 杩欎袱寮犲浘鐨勫彛寰勬槸锛?  - 宸︿晶鍥哄畾涓衡€滄棫 Strawberry + old hybrid-transformer鈥?  - 鍙充晶鍥哄畾涓衡€淎GC + current hybrid-transformer + joint_all鈥?  - 鐢ㄤ簬鍚戝甯堣鏄庯細鐪熸鍊煎緱璁茬殑涓嶆槸鈥滄棫缁撴瀯杩佸埌鏂版暟鎹泦鈥濓紝鑰屾槸鈥滄柊鏁版嵁闆嗚鏂扮殑 control-oriented hybrid-transformer 鍙樺緱鍚堢悊涓旀湁鏁堚€?- 宸叉柊澧炴贩鍚堟畫宸ā鍨嬭剼鏈細[benchmark_hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/benchmark_hybrid_residual_forecaster.py)
  - 瀹氫綅锛氫綔涓哄綋鍓嶉娴嬩富绾跨殑涓嬩竴姝ワ紝涓嶅啀缁х画鍫?plain Transformer锛岃€屾槸鎶?`DLinear` 鐨勭ǔ鍋ヤ富璺緞涓?`Transformer-hybrid` 鐨勯珮闃舵畫宸缓妯＄粨鍚堣捣鏉?  - 缁撴瀯锛歚ConditionalDLinearForecaster` 璐熻矗 main path锛宍ConditionalTransformerHybridForecaster` 璐熻矗 residual path锛屾渶缁堣緭鍑轰负 `base + gated residual`
  - 褰撳墠宸叉帴鍏?[main.py](c:/repositories/strawberry/agc_mpc/main.py) 鐨?baseline 鍏ュ彛锛屼篃鏀寔鐙珛 fair-budget benchmark
- 宸叉柊澧炰袱涓悓鍙ｅ緞 residual 鍊欓€夛細
  - `DLinear + iTransformer residual`
  - `DLinear + PatchTST residual`
- 宸叉柊澧炵粺涓€瀵规瘮鑴氭湰锛歔benchmark_residual_forecaster_variants.py](c:/repositories/strawberry/agc_mpc/benchmark_residual_forecaster_variants.py)
  - 褰撳墠涓夋潯鏈€浼樺厛棰勬祴閫夊瀷缁熶竴涓猴細
    - `transformer_hybrid_residual`
    - `itransformer_residual`
    - `patchtst_residual`
  - 鐩殑锛氬厛鍦ㄥ悓涓€ `fair-budget` 鍗忚涓嬫妸涓夋潯 residual 璺嚎鏀惧埌鍚屼竴鍙ｅ緞姣旇緝锛屽啀鍐冲畾璋佽繘鍏ユ帶鍒朵晶
- 宸插仛 1-epoch smoke test锛坄joint_all + Reference`锛宼argets = `Tair / Rhair / CO2air`锛夛細
  - [hybrid_residual_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/hybrid_residual_forecaster_joint_all_reference_summary.json)
  - [hybrid_residual_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/hybrid_residual_forecaster_joint_all_reference.pt)
  - `Tair`: Full `R2=0.8960`, MAE `0.912`; Final `R2=0.8904`, MAE `0.925`
  - `Rhair`: Full `R2=0.8828`, MAE `4.145`; Final `R2=0.8435`, MAE `4.887`
  - `CO2air`: Full `R2=0.6480`, MAE `58.135`; Final `R2=0.5861`, MAE `62.729`
- 宸茶ˉ鍋氬悓鍗忚 `DLinear` 1-epoch quick benchmark锛坄joint_all + Reference`锛宼argets = `Tair / Rhair / CO2air`锛夛細
  - [dlinear_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/dlinear_forecaster_joint_all_reference_summary.json)
  - [dlinear_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/dlinear_forecaster_joint_all_reference.pt)
  - `Tair`: Full `R2=0.8870`, MAE `1.003`; Final `R2=0.8745`, MAE `1.047`
  - `Rhair`: Full `R2=0.8872`, MAE `3.865`; Final `R2=0.8385`, MAE `4.651`
  - `CO2air`: Full `R2=0.5086`, MAE `71.191`; Final `R2=0.4850`, MAE `72.943`
- 宸叉柊澧炲揩閫熷姣斿浘锛?  - [hybrid_residual_vs_dlinear_joint_all_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/hybrid_residual_vs_dlinear_joint_all_reference.png)
  - 瀹氫綅锛氱敤浜庢槑澶╂眹鎶ユ椂蹇€熷睍绀衡€滃湪鍚屼竴 1-epoch quick benchmark 涓嬶紝娣峰悎娈嬪樊妯″瀷鐩稿 `DLinear` 鏄惁宸茬粡鍑虹幇鏃╂湡浼樺娍淇″彿鈥?- 褰撳墠璇绘硶锛?  - 杩欑粍缁撴灉浠呮槸 1-epoch smoke test锛屼笉鐢ㄤ簬姝ｅ紡缁撹
  - 浣嗗畠宸茬粡璇佹槑锛歚hybrid residual` 杩欐潯绾跨殑浠ｇ爜鍏ュ彛銆佽缁冦€乧heckpoint銆乻ummary 钀界洏閮藉凡鎵撻€氾紝鍙洿鎺ョ户缁窇姝ｅ紡棰勭畻
  - 鍦ㄥ綋鍓?1-epoch quick benchmark 涓嬶紝`hybrid residual` 宸茬粡鍦?`Tair / CO2air` 涓婃槑鏄句紭浜庡悓鍗忚 `DLinear`锛岃€?`Rhair` 涓庡叾鎺ヨ繎浣嗙暐閫?  - 鏇村悎鐞嗙殑涓嬩竴姝ユ槸涓?`current_hybrid_transformer` 浣跨敤鍚屼竴棰勭畻锛堝 `200 epoch, lr=1e-4, lambda_trend=0.3, patience=15`锛夊仛姝ｅ紡瀵规瘮锛屽啀鍐冲畾鏄惁鎺ュ叆鎺у埗渚?benchmark
- 宸茶ˉ鍋氬彟澶栦袱鏉?residual 鍊欓€夌殑鍚屽彛寰?1-epoch smoke test锛坄joint_all + Reference`锛宼argets = `Tair / Rhair / CO2air`锛夛細
  - [itransformer_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/itransformer_residual_joint_all_reference_summary.json)
  - [patchtst_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/patchtst_residual_joint_all_reference_summary.json)
  - `iTransformer residual`
    - `Tair`: Full `R2=0.8447`, MAE `1.108`; Final `R2=0.8308`, MAE `1.141`
    - `Rhair`: Full `R2=0.8793`, MAE `4.249`; Final `R2=0.8359`, MAE `4.994`
    - `CO2air`: Full `R2=0.6084`, MAE `60.546`; Final `R2=0.5862`, MAE `61.666`
  - `PatchTST residual`
    - `Tair`: Full `R2=0.9244`, MAE `0.729`; Final `R2=0.9131`, MAE `0.783`
    - `Rhair`: Full `R2=0.8816`, MAE `4.004`; Final `R2=0.8619`, MAE `4.555`
    - `CO2air`: Full `R2=0.6422`, MAE `57.291`; Final `R2=0.6286`, MAE `58.864`
- 褰撳墠涓夋潯 residual 璺嚎鍦?1-epoch smoke test 涓嬬殑鏃╂湡璇绘硶锛?  - `Transformer-hybrid residual` 浠嶆槸褰撳墠鏈€寮烘棭鏈熶俊鍙凤紝灏ゅ叾鍦?`Tair / CO2air` 涓婃渶鏄庢樉
  - `PatchTST residual` 鏄洰鍓嶆洿鍊煎緱缁х画鐨勭浜屽€欓€夛紝鏁翠綋宸茬粡鏄庢樉浼樹簬鍚屽崗璁?`DLinear`锛屼笖姣?`iTransformer residual` 鏇寸ǔ
  - `iTransformer residual` 鍏ュ彛宸叉墦閫氾紝浣嗗綋鍓?1-epoch 淇″彿鍋忓急锛屾殏鏃朵笉搴旀帓鍦ㄥ墠涓ゆ潯涔嬪墠
  - 鍥犳杩欏懆鏇村悎鐞嗙殑姝ｅ紡棰勭畻鎺ㄨ繘椤哄簭搴斾负锛?    - 鍏堣窇 `Transformer-hybrid residual`
    - 鍐嶈窇 `PatchTST residual`
    - `iTransformer residual` 鏆備繚鐣欎负绗笁鍊欓€?- `iTransformer residual` 宸插畬鎴?`joint_all + Reference + 200 epoch fair-budget` 姝ｅ紡杩愯锛?  - [itransformer_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/itransformer_residual_joint_all_reference_summary.json)
  - [itransformer_residual_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/itransformer_residual_joint_all_reference.pt)
  - `Tair`: Full `R2=0.9494`, MAE `0.618`; Final `R2=0.9362`, MAE `0.693`
  - `Rhair`: Full `R2=0.9030`, MAE `3.802`; Final `R2=0.8746`, MAE `4.412`
  - `CO2air`: Full `R2=0.7078`, MAE `51.161`; Final `R2=0.6947`, MAE `52.014`
- 褰撳墠瀵?`iTransformer residual` 姝ｅ紡缁撴灉鐨勮娉曪細
  - 瀹冨拰 1-epoch 缁撴灉鐩告瘮鎻愬崌闈炲父鏄庢樉锛岃鏄庤繖鏉＄嚎鏇翠緷璧栨寮忚缁冮绠楋紝涓嶈兘鐢ㄦ棭鏈?smoke test 杩囨棭鍚﹀畾
  - 鍦ㄥ綋鍓嶆寮忎笁鏉?residual 涓紝`iTransformer residual` 鏄渶鍧囪　鐨勪竴鏉★細
    - `Rhair` 鏈€寮?    - `CO2air` 涔熶紭浜庡彟澶栦袱鏉?    - `Tair` 铏戒笉濡?`Transformer-hybrid residual`锛屼絾浠嶄繚鎸佽緝寮?  - 鍥犳褰撳墠鏇寸ǔ鐨勬寮忕粨璁哄簲鏇存柊涓猴細
    - `Transformer-hybrid residual` = 娓╁害鏈€寮?    - `PatchTST residual` = 娆′紭涓斿 `CO2air` 鏈夋敼鍠?    - `iTransformer residual` = 褰撳墠涓夌洰鏍囨暣浣撴渶鍧囪　銆佹渶鍊煎緱浼樺厛鎺ュ叆鎺у埗渚ч獙璇佺殑 residual 鍊欓€?- `Transformer-hybrid residual` 宸插畬鎴?`joint_all + Reference + 200 epoch fair-budget` 姝ｅ紡杩愯锛?  - [transformer_hybrid_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/transformer_hybrid_residual_joint_all_reference_summary.json)
  - [transformer_hybrid_residual_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/transformer_hybrid_residual_joint_all_reference.pt)
  - `Tair`: Full `R2=0.9580`, MAE `0.526`; Final `R2=0.9502`, MAE `0.579`
  - `Rhair`: Full `R2=0.8210`, MAE `4.744`; Final `R2=0.7400`, MAE `5.945`
  - `CO2air`: Full `R2=0.6740`, MAE `55.315`; Final `R2=0.6310`, MAE `59.436`
- 褰撳墠瀵硅繖缁勬寮忕粨鏋滅殑璇绘硶锛?  - `Tair` 鏄庢樉鍙樺己锛屽凡缁忎紭浜?1-epoch smoke test锛屼篃浼樹簬鍚岀粨鏋勭殑鏃╂湡 quick benchmark
  - 浣?`Rhair / CO2air` 娌℃湁鍚屾鍙樻垚寮洪」锛岃鏄庡綋鍓?`Transformer-hybrid residual` 鐨勬寮忚缁冩敹鐩婁富瑕侀泦涓湪娓╁害涓荤洰鏍?  - 鍥犳瀹冧粛鐒舵槸褰撳墠 residual 涓荤嚎鐨勫己鍊欓€夛紝浣嗚繕涓嶈兘鐩存帴璁ゅ畾涓衡€滀笁鐩爣鏁翠綋鏈€浼樷€?  - 涓嬩竴姝ョ户缁寮忚窇 `PatchTST residual` 浠嶇劧鏄繀瑕佺殑锛屽洜涓哄畠鍦?1-epoch 涓嬬殑 `Rhair` 淇″彿鏇存帴杩戝彲绔炰簤鍖洪棿
- `PatchTST residual` 宸插畬鎴?`joint_all + Reference + 200 epoch fair-budget` 姝ｅ紡杩愯锛?  - [patchtst_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/patchtst_residual_joint_all_reference_summary.json)
  - [patchtst_residual_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/patchtst_residual_joint_all_reference.pt)
  - `Tair`: Full `R2=0.9440`, MAE `0.676`; Final `R2=0.9230`, MAE `0.829`
  - `Rhair`: Full `R2=0.8468`, MAE `4.991`; Final `R2=0.8121`, MAE `5.780`
  - `CO2air`: Full `R2=0.7311`, MAE `46.962`; Final `R2=0.6376`, MAE `55.862`
- 褰撳墠瀵?`PatchTST residual` 姝ｅ紡缁撴灉鐨勮娉曪細
  - 鐩告瘮鍏?1-epoch smoke test锛屾寮忚缁冨悗 `CO2air` 鏄庢樉鍙樺己锛屾垚涓哄綋鍓嶈繖鏉＄嚎鏈€绐佸嚭鐨勬敹鐩婄偣
  - `Rhair` 涔熶紭浜庡綋鍓嶆寮忕増 `Transformer-hybrid residual`
  - 浣?`Tair` 浠嶆槑鏄惧急浜庡綋鍓嶆寮忕増 `Transformer-hybrid residual`
  - 鍥犳褰撳墠鏇寸ǔ鐨勭粨璁轰笉鏄€滃摢涓€鏉″叏璧⑩€濓紝鑰屾槸锛?    - `Transformer-hybrid residual` 鏇存搮闀?`Tair`
    - `PatchTST residual` 鏇存搮闀?`Rhair / CO2air`
  - 鍦?`iTransformer residual` 姝ｅ紡缁撴灉鍑烘潵鍚庯紝`PatchTST residual` 鏇村噯纭殑瀹氫綅搴旇皟鏁翠负锛?    - 瀹冧粛鐒舵槸鍚堢悊鐨勭浜屾闃熷€欓€?    - 浣嗗綋鍓嶆暣浣撳潎琛℃€т笉濡?`iTransformer residual`
  - 杩欒繘涓€姝ユ敮鎸佸綋鍓嶄富绾垮垽鏂細瀵?AGC 杩欑被澶氱洰鏍囨帶鍒跺鍚戜换鍔★紝涓嶅瓨鍦ㄥ崟涓€缁濆鏈€浼樼粨鏋勶紝寮烘ā鍨嬪彲鑳芥寜鐩爣鍙橀噺鍒嗗寲
- 宸叉柊澧炵粺涓€ residual 鍑哄浘鑴氭湰锛歔plot_residual_forecaster_variants.py](c:/repositories/strawberry/agc_mpc/plot_residual_forecaster_variants.py)
  - 浣滅敤锛氬鐢ㄥ師濮?forecasting evaluator 鐨勭敾鍥鹃摼璺紝涓?residual 姝ｅ紡妯″瀷琛ラ綈涓?baseline 鐩稿悓鏍煎紡鐨勪笁绫诲浘锛?    - `forecast_examples`
    - `forecast_rollout`
    - `horizon_mae`
- 褰撳墠涓夋潯 residual 姝ｅ紡妯″瀷鐨勫浘鏂囦欢宸插叏閮ㄧ敓鎴愶細
  - `Transformer-hybrid residual`
    - [transformer_hybrid_residual_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/transformer_hybrid_residual_joint_all_reference_forecast_examples.png)
    - [transformer_hybrid_residual_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/transformer_hybrid_residual_joint_all_reference_forecast_rollout.png)
    - [transformer_hybrid_residual_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/transformer_hybrid_residual_joint_all_reference_horizon_mae.png)
  - `iTransformer residual`
    - [itransformer_residual_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/itransformer_residual_joint_all_reference_forecast_examples.png)
    - [itransformer_residual_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/itransformer_residual_joint_all_reference_forecast_rollout.png)
    - [itransformer_residual_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/itransformer_residual_joint_all_reference_horizon_mae.png)
  - `PatchTST residual`
    - [patchtst_residual_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/patchtst_residual_joint_all_reference_forecast_examples.png)
    - [patchtst_residual_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/patchtst_residual_joint_all_reference_forecast_rollout.png)
    - [patchtst_residual_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/residual_variants/patchtst_residual_joint_all_reference_horizon_mae.png)
- 宸叉柊澧炴爣鍑嗗嚭鍥捐剼鏈細[plot_current_hybrid_transformer_standard.py](c:/repositories/strawberry/agc_mpc/plot_current_hybrid_transformer_standard.py)
  - 浣滅敤锛氫负 `current_hybrid_transformer` 澶嶇敤缁熶竴 evaluator锛岃ˉ榻愪笌 baseline / residual 鐩稿悓鏍煎紡鐨勪笁绫诲浘锛屽苟鎶?`figure_paths` 鍥炲啓鍒?summary
- `current_hybrid_transformer` 鏍囧噯涓夊浘鐜板凡琛ラ綈锛?  - `horizon=24`
    - [current_hybrid_transformer_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_joint_all_reference_forecast_examples.png)
    - [current_hybrid_transformer_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_joint_all_reference_forecast_rollout.png)
    - [current_hybrid_transformer_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_joint_all_reference_horizon_mae.png)
  - `horizon=120`
    - [current_hybrid_transformer_h120_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h120_joint_all_reference_forecast_examples.png)
    - [current_hybrid_transformer_h120_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h120_joint_all_reference_forecast_rollout.png)
    - [current_hybrid_transformer_h120_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h120_joint_all_reference_horizon_mae.png)


