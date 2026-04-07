# CONTEXT.md

## 0. 鐎电鐦芥稉搴ゃ€冩潏鎹愵潐閸?
- 姒涙顓婚幎濠傛礀缁涙柨顕挒陇顫嬫稉琛♀偓婊€绮犻梿璺虹磻婵绨＄憴锝夈€嶉惄顔炬畱娴滆　鈧縿鈧?- 鐟欙綁鍣村鍌氬悍閺冩湹绱崗鍫滃▏閻劋鑵戦弬鍥モ偓?- 闂勩倝娼箛鍛淬€忛敍灞肩瑝鐟曚椒鑵戦懟鍗炪仚閺夊倶鈧?- 婵″倹鐏夎箛鍛淬€忔担璺ㄦ暏閼昏鲸鏋冮張顖濐嚔閿涘矁顩︾粩瀣煝閸︺劌鎮楅棃銏Ｋ夋稉顓熸瀮闁插﹣绠熼敍灞肩伐婵″倵鈧粌閽╅崸鍥╃卜鐎电顕ゅ顕嗙礄Mean Absolute Error閿涘AE閿涘鈧縿鈧?- 鐠佹彃娴橀妴浣筋唹閹稿洦鐖ｉ妴浣筋唹濡€崇€烽弮璁圭礉閸忓牐顕╅垾婊冪暊閺勵垯绮堟稊鍫氣偓婵撶礉閸愬秷顕╅垾婊冪暊鐠囧瓨妲戞禍鍡曠矆娑斿牃鈧繐绱濋張鈧崥搴″晙鐠囩补鈧粍鈧簼绠炵憴锝堫嚢閳ユ縿鈧?- 濞戝寮风€硅妲楀ǎ閿嬬┋閻ㄥ嫭婀崇拠顓熸閿涘奔绱崗鍫㈢舶閸戣櫣娲块惂鍊熜掗柌濠忕礉娑撳秹绮拋銈呬海鐠佹崘顕伴懓鍛嚒閺堝婧€閸ｃ劌顒熸稊鐘冲灗閹貉冨煑閼冲本娅欓妴?- forecasting 姒涙顓荤紒鎾寸亯閸ユ儳褰ф穱婵堟殌娑撳琚敍?  - `forecast_examples`閿涙艾宕熷▎锟狀暕濞村鐗辨笟瀣禈
  - `forecast_rollout`閿涙碍绮撮崝銊ヮ樋缁愭顣╁ù瀣禈
  - `horizon_mae`閿涙岸顣╁ù瀣劄闂€鑳嚖瀹割喖娴?- `forecast_error_heatmap` 瀹歌尙些闂勩倧绱濇稉宥呭晙娴ｆ粈璐熸妯款吇缂佹挻鐏夐崶鎹愮翻閸戞亽鈧?- `forecast_first_step_rollout` 娑旂喎鍑＄粔濠氭珟閿涘奔绗夐崘宥勭稊娑撴椽绮拋銈囩波閺嬫粌娴樻潏鎾冲毉閵?
## 1. 娴ｈ法鏁ら弬鐟扮础

鏉╂瑦妲搁張顒勩€嶉惄顔炬畱闂€鎸庢埂娑撳﹣绗呴弬鍥ㄦ瀮娴犺翰鈧?
瀵ら缚顔呯憴鍕灟閿?
1. 濮ｅ繑顐煎鈧崥顖涙煀鐎电鐦介弮璁圭礉娴兼ê鍘涚拠璇插絿閺堫剚鏋冩禒韬测偓?2. 婵″倹鐏夋担璺ㄦ暏閺€顖涘瘮閺傚洣娆㈡稉濠佺瑓閺傚洨娈?IDE / AI 閸斺晜澧滈敍宀€娲块幒銉ョ穿閻劍婀伴弬鍥︽閵?3. 濮ｅ繑顐肩€瑰本鍨氶張澶嬪壈娑斿娈戞禒锝囩垳閺€鐟板З閵嗕礁鐤勬宀€绮ㄩ弸婊勬纯閺傝埇鈧浇鐭剧痪鑳殶閺佹潙鎮楅敍宀勫厴鐟曚焦娲块弬鐗堟拱閺傚洣娆㈤妴?4. 閺堫剚鏋冩禒鏈电喘閸忓牐顔囪ぐ鏇犌旂€规矮绨ㄧ€圭偑鈧礁缍嬮崜宥勫瘜缁捐￥鈧礁鍙ч柨顔煎枀缁涙牓鈧焦娓堕弬鎵波閺嬫嚎鈧箑ODO 閸滃苯浼愭担婊嗩潐閸掓瑣鈧?

## 2. 妞ゅ湱娲拌ぐ鎾冲娑撹崵鍤?
瑜版挸澧犻惄顔界垼娑撳秵妲告径宥囧箛閸樼喕宕忛懢鎾诡啈閺傚浄绱濋懓灞炬Ц閸嬫熬绱?
**闂堛垹鎮滈幒褍鍩楅惃鍕刊鐎广倕顦垮銉╊暕濞?+ 闂傤厾骞?MPC**

閺嶇绺剧拋鎯х暰閿?
- 娴ｈ法鏁ゆ径姘綁闁插繑淇€广倖鏆熼幑?- 鏉堟挸鍙嗛崢鍡楀蕉鐎广倕鍞撮悩鑸碘偓?- 鏉堟挸鍙嗛張顏呮降婢垛晜鐨?/ 婢舵牜鏁撻柌?- 鏉堟挸鍙嗛張顏呮降閹貉冨煑鐠佹儳鐣鹃崐?- 妫板嫭绁撮張顏呮降鐎广倕鍞村〒鈺侇吇閻樿埖鈧?- 閺堚偓缂佸牊婀囬崝鈥茬艾 MPC
- SAC 娴犲懍缍旀稉?baseline閿涘奔绗夐弰顖欏瘜缁炬寧鏌熷▔?

## 3. 瑜版挸澧犳い鍦窗閸掑棗浼?
### 3.1 閺冄囥€嶉惄?
- [diffmpc](c:/repositories/strawberry/diffmpc)

鐠囧瓨妲戦敍?
- 鏉╂瑦妲搁弮褑宕忛懢鎾插瘜缁惧潡銆嶉惄?- 娑撳秴鍟€娴ｆ粈璐熼弬鎵畱娑撴槒顩﹂拃钘夋勾閺傜懓鎮?- 娴犲懍绻氶悾娆忓棘閼板啩鐜崐?
### 3.2 閺傞瀵屾い鍦窗

- [agc_mpc](c:/repositories/strawberry/agc_mpc)

鐠囧瓨妲戦敍?
- 鏉╂瑦妲搁弬鎵畱 AGC 2019 娑撹崵鍤庢い鍦窗
- 閸氬海鐢绘稉鏄忣洣娴狅絿鐖滃銉ょ稊闁棄绨叉导妯哄帥閺€鎯ф躬鏉╂瑩鍣?- 閸樼喎鍨稉濠佺瑝鐟曚礁鍟€閹跺﹥鏌婂鈧崣鎴犳埛缂侇厼鐖㈤崶?`diffmpc`


## 4. 瑜版挸澧犻弽绋跨妇閺佺増宓侀梿?
### 4.1 娑撶粯鏆熼幑顕€娉?
- [AutonomousGreenhouseChallenge_edition2](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2)

鏉╂瑦妲歌ぐ鎾冲娑撹缂撳Ο鈩冩殶閹诡喗绨妴?
### 4.2 閸樼喎顫愰崠?/ 婢跺洣鍞?
- [Autonomous Greenhouse Challenge, Second Edition (2019)_1_all](c:/repositories/strawberry/Autonomous%20Greenhouse%20Challenge,%20Second%20Edition%20(2019)_1_all)

鐠囧瓨妲戦敍?
- 鏉╂瑦妲搁崢鐔奉潗娑撳娴囬崠鍛挤閸忔儼袙閸樺鎮楅惃鍕槵娴犵晫绮ㄩ弸?- 娑撳秳缍旀稉杞板瘜瀵ょ儤膩閸忋儱褰?- 娴犲懎婀棁鈧憰浣告礀閺屻儱甯慨瀣壐瀵繑妞傛担璺ㄦ暏

### 4.3 閼藉甯楅弫鐗堝祦

- [Strawberry Greenhouse Environmental Control Dataset(version2).csv](c:/repositories/strawberry/Strawberry%20Greenhouse%20Environmental%20Control%20Dataset(version2).csv)

鐠囧瓨妲戦敍?
- 閻滄澘婀梽宥囬獓娑?secondary dataset / stress test
- 娑撳秴鍟€娴ｆ粈璐熺拋鐑樻瀮娑撹鐤勬灞炬殶閹诡噣娉?

## 5. AGC 閺佺増宓侀悶鍡毿?
閺夊啫鈻夐崣鍌濃偓鍐跨窗

- [ReadMe.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/ReadMe.pdf)
- [Economics.pdf](c:/repositories/strawberry/AutonomousGreenhouseChallenge_edition2/Economics.pdf)
- [AGC_DATA_SCHEMA.md](c:/repositories/strawberry/AGC_DATA_SCHEMA.md)

閸忔娊鏁紒鎾诡啈閿?
- `Weather.csv` = 閺堫亝娼垫径鏍晸婢垛晜鐨?- `GreenhouseClimate.csv` = 鐎广倕鍞撮悩鑸碘偓?+ 閹笛嗩攽閸ｃ劎濮搁幀?+ 鐠佹儳鐣鹃崐?- `*_sp` = 鐠囬攱鐪扮拋鎯х暰閸?- `*_vip` = realized setpoints
- `Resources.csv` = 閺冦儱鏄傛惔锕佺カ濠ф劖绉烽懓?- `Production.csv` = 閺€鎯板箯閺冨墎鍋ｆ禍褔鍣?- `TomQuality.csv` = 閸濅浇宸?- `GrodanSens.csv` = 閺嶇懓灏?/ 閸╅缚宸濋弫鐗堝祦

瑜版挸澧犵粭顑跨閻楀牆缂撳Ο鈩冨复閸欙綇绱?
- `x_past`: 閸樺棗褰剁€广倕鍞撮悩鑸碘偓?+ 閹笛嗩攽閸ｃ劌寮芥＃?- `w_future`: 婢垛晜鐨?+ 閺冨爼妫块悧鐟扮窙
- `u_future`: 閺堫亝娼?setpoints
- `y_future`: 閺堫亝娼?`Tair / Rhair / CO2air / Tot_PAR`


## 6. 瑜版挸澧犻弬鏉夸紣缁嬪鍞惍浣哄Ц閹?
妞ゅ湱娲伴惄顔肩秿閿?
- [agc_mpc](c:/repositories/strawberry/agc_mpc)

閺嶇绺鹃弬鍥︽閿?
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

瑜版挸澧犲鎻掔暚閹存劧绱?
- AGC 閺佺増宓佺拠璇插絿
- 閺冨爼妫跨€涙顔岄弽鍥у櫙閸?- 婢垛晜鐨垫稉搴㈢毜閸婃瑨銆冪€靛綊缍?- `sp/vip` 缂傚搫銇戦崶鐐诧綖
- `x_past / w_future / u_future / y_future` 閺嶉攱婀伴崚鍥╁
- 閸楁洟娈ч梻鏉戞嫲婢舵岸娈ч梻瀛樻暜閹?- 閸忋劌鐪?leak-free 閺冭泛绨崚鍥у瀻
- 婢舵岸娈ч梻纾嬩粓閸氬牐顔勭紒鍐х瑓閻ㄥ嫬鍙忕仦鈧弽鍥у櫙閸?- 閺夆€叉 GRU baseline
- 閺夆€叉 DLinear baseline
- 閺夆€叉 SegRNN baseline
- 閺夆€叉缁?Transformer baseline
- 閺夆€叉 Transformer-hybrid baseline
- `DLinear main path + Transformer-hybrid residual` 濞ｅ嘲鎮庡▓瀣▕濡€崇€烽崢鐔风€?- 缁傝崵鍤庣拠鍕強鏉堟挸鍤?- forecast 閸ョ偓鏁幐浣测偓婊冨坊閸欒弓绗傛稉瀣瀮 + 閺堫亝娼?horizon閳ユ繆浠堥崥鍫濈潔缁€鐚寸礉娑撳秴鍟€閸欘亞娲撮惈鈧痪?future window
- forecast 閸ョ偓鏌婃晶?rolling multi-step rollout 鐏炴洜銇氶敍宀€鏁ら弴鎾毐閺冨爼妫挎潪瀛樻▔缁€楦跨箾缂侇厼顦跨粣妤咁暕濞村绱濋懓灞肩瑝閸欘亝妲告稉鈧▓?24-step future window
- forecasting 閻滄澘婀妯款吇娣囨繄鏆€ 3 缁顣╁ù瀣禈閿涙艾宕熷▎锟狀暕濞村鐗辨笟瀣禈閵嗕焦绮撮崝銊ヮ樋缁愭顣╁ù瀣禈閵嗕線顣╁ù瀣劄闂€鑳嚖瀹割喖娴?- 娑撳﹨鍫?3 缁鏆遍弮鍫曟？鏉炴潙娴樺鑼病娑?`GRU / DLinear / SegRNN / Transformer / Transformer-hybrid` 閸忋劑鍎寸悰銉╃秷
- `results` 閻╊喖缍嶅鈧慨瀣瘻 `forecasting / control` 閸掑棗鐪伴弫瀵告倞
- forecasting checkpoint 缂佺喍绔撮弨鑸垫殐閸?`agc_mpc/results/forecasting/checkpoints`
- forecasting 閸ュ墽绮烘稉鈧弨鑸垫殐閸?`agc_mpc/results/forecasting/figures`
- control summary 缂佺喍绔撮弨鑸垫殐閸?`agc_mpc/results/control/summaries`
- AGC 閹貉冨煑娓氀冨灥閻楀牊甯撮崗?- `DLinear / Transformer-hybrid` 瀹稿弶甯撮崚?AGC 娑撳﹦娈戞稉銈囪 MPC 濮瑰倽袙閸?- `CEMMPC` 瀹歌尪藟娑撳﹤娴愮€规岸娈㈤張铏诡潚鐎涙劑鈧簚arm start閵嗕恭andidate injection 閸滃本娲块獮铏拨閻?CEM 閺囧瓨鏌?- 闂傤厾骞?rollout 姒涙顓婚崚鍥у煂閺囩繝寮楅弽鑲╂畱 `surrogate` 濡€崇础閿涘奔绗夐崘宥夌帛鐠併倗鏁ら惇鐔风杽娑撳绔寸悰宀€濮搁幀浣瑰ⅵ鎼?- surrogate 閻樿埖鈧焦娲块弬浼村櫡娴兼岸鍣哥粻?`HumDef`閿涘苯鑻熼悽?persistence + action proxy 閺囧瓨鏌婇棃鐐垫窗閺嶅洨濮搁幀?- 閹貉冨煑缂佹挻鐏夐懛顏勫З娣囨繂鐡ㄩ崚?`agc_mpc/results/control`
- 瀹稿弶鏌婃晶?`benchmark_hybrid_residual_forecaster.py`閿涘瞼鏁ゆ禍搴℃躬閸忣剙閽╃拋顓犵矊妫板嫮鐣绘稉瀣礋閻欘剝鐦庢导鐗堣穿閸氬牊鐣顔侥侀崹?
瑜版挸澧犻張顏勭暚閹存劧绱?
- 鐎瑰本鏆ｉ悧鈺冩倞缁?/ economic 缁?AGC 闂傤厾骞嗛悳顖氼暔
- 閺囩繝寮楅弽鑲╂畱 actuator / VIP / resource-aware AGC 閹貉冨煑瀵ょ儤膩
- 鐠у嫭绨幋鎰拱 / 缂佸繑绁归幐鍥ㄧ垼缁惧啿鍙嗛幒褍鍩楅惄顔界垼


## 7. 瑜版挸澧犳妯款吇鐎圭偤鐛欑拋鍓х枂

閺夈儴鍤?[config.py](c:/repositories/strawberry/agc_mpc/config.py)閿?
- 姒涙顓婚梾鏃堟？閿? 娑擃亜鍙忛柈銊ㄤ粓閸氬牐顔勭紒?- `seq_len = 288`  
  閸氼偂绠熼敍?4 鐏忓繑妞傞崢鍡楀蕉缁愭褰?- `horizon = 24`  
  閸氼偂绠熼敍? 鐏忓繑妞傛０鍕ゴ缁愭褰?- 鏉╂瑦鍓伴崨宕囨絻閳ユ粌宕熸稉?forecast 缁愭褰涢崶閿偓婵嗐亯閻掕泛褰ф导姘▔缁€?24 娑擃亝婀弶銉︻劄閿涙稑顩ч弸婊勫厒閻娲块梹鎸庢闂傜閰遍敍宀勬付鐟曚胶婀?rolling forecast rollout 閸ユ拝绱濋幋鏍纯閹恒儲濡?`horizon` 閺€鐟般亣閸氬酣鍣哥拋?- `batch_size = 256`
- `num_epochs = 12`
- `early_stop_patience = 4`
- `control_eval_steps = 96`
- `control_rollout_mode = surrogate`

瑜版挸澧犻惄顔界垼閸欐﹢鍣洪敍?
- `Tair`
- `Rhair`
- `CO2air`
- `Tot_PAR`


## 8. 閺堚偓閺傛澘鐔€缁捐法绮ㄩ弸?
閺堚偓閺傛媽绻嶇悰灞炬煙瀵骏绱?
```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\main.py
```

閺堚偓閺傜増鏆熼幑顔款潐濡槄绱?
- 6 娑擃亪娈ч梻纾嬩粓閸氬牐顔勭紒?- `train = 199488`
- `val = 40878`
- `test = 40878`

### 8.1 GRU baseline

- `Tair`: Full `R铏?0.9293`, MAE `0.886`; Final `R铏?0.9136`, MAE `1.026`
- `Rhair`: Full `R铏?0.8277`, MAE `3.996`; Final `R铏?0.7424`, MAE `5.067`
- `CO2air`: Full `R铏?0.7718`, MAE `55.797`; Final `R铏?0.7092`, MAE `64.391`
- `Tot_PAR`: Full `R铏?0.9688`, MAE `37.947`; Final `R铏?0.9660`, MAE `39.784`

缂佹挻鐏夐崶鎾呯窗

- [gru_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/gru_baseline_forecast_examples.png)
- [gru_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/gru_baseline_horizon_mae.png)

### 8.2 DLinear baseline

- `Tair`: Full `R铏?0.9639`, MAE `0.638`; Final `R铏?0.9526`, MAE `0.729`
- `Rhair`: Full `R铏?0.8607`, MAE `3.684`; Final `R铏?0.8184`, MAE `4.209`
- `CO2air`: Full `R铏?0.8205`, MAE `48.084`; Final `R铏?0.7928`, MAE `51.481`
- `Tot_PAR`: Full `R铏?0.9790`, MAE `30.483`; Final `R铏?0.9779`, MAE `31.295`

缂佹挻鐏夐崶鎾呯窗

- [dlinear_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_baseline_forecast_examples.png)
- [dlinear_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_baseline_horizon_mae.png)

### 8.3 SegRNN baseline

- `Tair`: Full `R铏?0.9228`, MAE `0.944`; Final `R铏?0.9076`, MAE `1.069`
- `Rhair`: Full `R铏?0.7512`, MAE `4.951`; Final `R铏?0.6662`, MAE `5.922`
- `CO2air`: Full `R铏?0.7856`, MAE `53.093`; Final `R铏?0.7176`, MAE `62.168`
- `Tot_PAR`: Full `R铏?0.9689`, MAE `38.705`; Final `R铏?0.9672`, MAE `40.208`

缂佹挻鐏夐崶鎾呯窗

- [segrnn_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/segrnn_baseline_forecast_examples.png)
- [segrnn_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/segrnn_baseline_horizon_mae.png)

### 8.4 缁?Transformer baseline

- `Tair`: Full `R铏?0.9483`, MAE `0.765`; Final `R铏?0.9413`, MAE `0.823`
- `Rhair`: Full `R铏?0.8038`, MAE `4.249`; Final `R铏?0.7454`, MAE `4.919`
- `CO2air`: Full `R铏?0.8509`, MAE `43.206`; Final `R铏?0.8242`, MAE `47.229`
- `Tot_PAR`: Full `R铏?0.9853`, MAE `26.484`; Final `R铏?0.9859`, MAE `24.964`

缂佹挻鐏夐崶鎾呯窗

- [transformer_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_baseline_forecast_examples.png)
- [transformer_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_baseline_horizon_mae.png)

### 8.5 Transformer-hybrid baseline

- `Tair`: Full `R铏?0.9544`, MAE `0.708`; Final `R铏?0.9480`, MAE `0.770`
- `Rhair`: Full `R铏?0.7539`, MAE `4.650`; Final `R铏?0.6927`, MAE `5.306`
- `CO2air`: Full `R铏?0.7870`, MAE `51.905`; Final `R铏?0.7434`, MAE `58.318`
- `Tot_PAR`: Full `R铏?0.9848`, MAE `28.237`; Final `R铏?0.9846`, MAE `28.509`

缂佹挻鐏夐崶鎾呯窗

- [transformer_hybrid_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_hybrid_baseline_forecast_examples.png)
- [transformer_hybrid_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/transformer_hybrid_baseline_horizon_mae.png)

瑜版挸澧犵粋鑽ゅ殠缂佹捁顔戦敍?
- `DLinear` 娴犲秶鍔ч弰?`Tair / Rhair` 娑撳﹥娓剁粙宕囨畱閺佺繝缍?baseline
- 缁?`Transformer` 閸︺劌缍嬮崜宥堫啎缂冾喕绗呯€?`CO2air / Tot_PAR` 閺堚偓瀵尨绱濇稉鏃€鏆ｆ担鎾冲繁娴滃骸缍嬮崜?`Transformer-hybrid`
- `Transformer-hybrid` 娴犲秳绻氶悾娆戠波閺嬪嫪鐜崐纭风礉娴ｅ棗缍嬮崜宥呯杽閻滅増鐥呴張澶婃躬閹碘偓閺堝娲伴弽鍥︾瑐鐡掑懓绻冪痪?Transformer
- `GRU` 瑜版挸澧犳稉宥呭晙閺勵垱鏆ｆ担鎾存付娴兼﹫绱濇担鍡曠矝閻掕埖妲搁柌宥堫洣閻ㄥ嫭妞傛惔?baseline
- `SegRNN` 瑜版挸澧犻張顏囩Т鏉╁洤澧犳稉澶庘偓?- 鏉╂瑧鎴风紒顓熸暜閹镐椒绔存稉顏堝櫢鐟曚浇顔戦弬鍥啈閻愮櫢绱? 
  **閺堚偓婵傜晫娈戠粋鑽ゅ殠妫板嫭绁村Ο鈥崇€烽崣顖濆厴閸ョ姷娲伴弽鍥у綁闁插繋绗夐崥宀冣偓灞藉瀻閸栨牭绱濇稉宥呯摠閸︺劌宕熸稉鈧紒婵嗩嚠閺堚偓娴兼绮ㄩ弸?*

### 8.6 閹貉冨煑娓?benchmark閿?026-03-23, stricter surrogate update閿?
鏉╂劘顢戦弬鐟扮础閿?
```bash
conda activate strawberry_env
python c:\repositories\strawberry\agc_mpc\control_main.py --steps 48 --start-idx 0 --reference-mode trajectory
```

閸楀繗顔呯拠瀛樻閿?
- 閹貉冨煑闂呮棃妫块敍姝歊eference`
- 閹貉冨煑閸ｎ煉绱癭recorded` / `GradientMPC` / `CEMMPC`
- 妫板嫭绁撮崳顭掔窗`DLinear`閵嗕胶鍑?`Transformer`閵嗕梗Transformer-hybrid`
- 閸欏倽鈧啰娲伴弽鍥风窗濞村鐦梿鍡欐埂鐎圭偞婀弶?`y_future` trajectory
- 瑜版挸澧犻梻顓犲箚閸楀繗顔呮禒宥勭瑝閺勵垰鐣弫瀵稿⒖閻炲棔璞㈤惇鐔锋珤閿涘奔绲惧В鏂剧瑐娑撯偓閻楀牊娲挎稉銉︾壐閿?  - 婢垛晜鐨甸妴浣规闂傛潙鎷伴崣鍌濃偓鍐缓鏉╁湱鎴风紒顓熸降閼?AGC 濞村鐦梿鍡欐埂鐎圭偛绨崚?  - 鐞氼偅甯堕惄顔界垼閻樿埖鈧胶鏁辨０鍕ゴ閸ｃ劋绔村銉︾泊閸斻劋楠囬悽?  - 閸樺棗褰堕悩鑸碘偓渚€绮拋銈勭瑝閸愬秶娲块幒銉﹀鐠愭繄婀＄€圭偘绗呮稉鈧悰宀嬬礉閼板本妲告禒搴＄秼閸撳秶濮搁幀浣稿毉閸欐埊绱濋悽?persistence + action proxy + predicted targets 閺囧瓨鏌?  - `HumDef` 閺嶈宓佹０鍕ゴ閻?`Tair / Rhair` 闁插秵鏌婄拋锛勭暬
  - `CEMMPC` 閻滄澘婀担璺ㄦ暏閸ュ搫鐣鹃梾蹇旀簚缁夊秴鐡欓敍灞芥礈濮濄倕鎮撴稉鈧崨鎴掓姢闁插秷绐囬弮?summary 閸濆牆绗囨穱婵囧瘮娑撯偓閼?
缂佹挻鐏夐敍?
閺堫垵顕㈢拠瀛樻閿?
- 鏉╂瑩鍣烽崢鐔告降閸愭瑤缍?`DPC` 閻ㄥ嫭鏌熷▔鏇礉閻滄澘婀紒鐔剁鐠侀璐?`GradientMPC`
- 鐎瑰啩绗夐弰顖滃缁斿绨?MPC 閻ㄥ嫬褰熸稉鈧猾缁樺付閸掓儼瀵栧蹇ョ礉閼板本妲搁垾婊堚偓姘崇箖濮婎垰瀹抽惄瀛樺复濮瑰倽袙濠婃艾濮╅弮璺虹厵娴兼ê瀵查梻顕€顣介惃?MPC 濮瑰倽袙閸ｃ劉鈧?- 閸樼喐娼甸崘娆庣稊 `MPC(CEM)` 閻ㄥ嫭鏌熷▔鏇礉閻滄澘婀紒鐔剁鐠侀璐?`CEMMPC`
- 閸ョ姵顒濊ぐ鎾冲閹貉冨煑鐎佃鐦弴鏉戝櫙绾喖婀寸拠瀛樻Ц閿涙瓪GradientMPC vs CEMMPC`

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

瑜版挸澧犻幒褍鍩楃紒鎾诡啈閿?
- 閸︺劍娲挎稉銉︾壐閻?48-step surrogate rollout 娑撳绱漙GradientMPC` 娴犲秵娅橀柆宥勭喘娴?`CEMMPC`
- `DLinear + GradientMPC` 閺勵垰缍嬮崜宥嗘付瀵櫣娈戞稉銉︾壐 surrogate 閹貉冨煑缂佸嫬鎮庨敍灞芥磽娑擃亞娲伴弽鍥厴閺勬崘鎲叉导妯圭艾 recorded
- `CEMMPC` 閻滄澘婀鑼病閸欘垰顦查悳甯礉閸氬奔绔撮崨鎴掓姢闁插秴顦叉潻鎰攽閺冭泛鍙?summary 閸濆牆绗囨穱婵囧瘮娑撯偓閼疯揪绱濇担鍡樷偓褑鍏樻禒宥堟儰閸氬簼绨?`GradientMPC`
- surrogate 閸楀繗顔呮稉鈧弮锔芥暪缁毖嶇礉recorded control 閸滃苯鎮?predictor 閻ㄥ嫯顕ゅ顕€鍏樻导姘閺勬儳褰夋径褝绱濇潻娆掝嚛閺勫簼绗傛稉鈧悧?semi-grounded rollout 绾喖鐤勯崑蹇庣鐟?- 鏉╂瑨绻樻稉鈧銉﹀絹缁€鐚寸窗**閺堚偓瀵櫣顬囩痪鍧楊暕濞村娅掓稉宥勭鐎规俺鍤滈崝銊ュ綁閹存劖娓跺娲４閻滎垱甯堕崚?surrogate**


## 9. 瑜版挸澧犻懓妤佹缂佸繘鐛欓崐?
閸︺劌缍嬮崜宥嗘簚閸ｃ劌鎷拌ぐ鎾冲闁板秶鐤嗘稉瀣剁窗

- 閸楁洟娈ч梻?GRU baseline閿涙氨瀹?`20 缁夋妶
- 6 闂呮棃妫块懕鏂挎値 `GRU + DLinear + SegRNN`閿涙氨瀹?`136 缁夋妶
- 6 闂呮棃妫块懕鏂挎値 `GRU + DLinear + SegRNN + Transformer-hybrid`閿涙氨瀹?`541 缁夋妶

缁鏆愭导鎷岊吀閿?
- 鏉炲鍣?baseline閿涙瓪2 閸掑棝鎸撻崘鍗?- 娑擃厾鐡戠憴鍕?GRU / SegRNN閿涙瓪2~5 閸掑棝鎸揱
- Transformer / hybrid閿涙瓪6~10 閸掑棝鎸揱


## 10. 瑜版挸澧犵拋鐑樻瀮鐎规矮缍?
瑜版挸澧犻張鈧导妯虹暰娴ｅ秳绗夐弰顖ょ窗

- 閳ユ粍鏁兼潻?Transformer 閸嬫碍淇€广倝顣╁ù瀣р偓?
閺囨潙鎮庨悶鍡欐畱鐎规矮缍呴弰顖ょ窗

- 閳ユ粓娼伴崥鎴炲付閸掑墎娈戝〒鈺侇吇婢舵碍顒炴０鍕ゴ閳?- 閳ユ粌鍩勯悽銊︽弓閺夈儱銇夊鏂剧瑢閺堫亝娼甸幒褍鍩楁穱鈩冧紖閻ㄥ嫰妫撮悳顖烆暕濞村甯堕崚鑸殿攱閺嬪灈鈧?- 閳ユ粓顣╁ù瀣侀崹瀣╃瑢閹貉冨煑閹嗗厴娑斿妫块崗宕囬兇閻ㄥ嫮閮寸紒?benchmark閳?

## 11. 瑜版挸澧犻崚娑欐煀閻愮懓鍨介弬?
### 閸欘垯浜掗幋鎰彌閻ㄥ嫬鍨遍弬鎵仯

- 闂堛垹鎮滈幒褍鍩楅惃鍕樋濮濄儵顣╁ù瀣剁礉閼板奔绗夐弰顖滃嚱缁傝崵鍤庨幏鐔锋値
- 閺勬儳绱￠崚鈺冩暏閺堫亝娼垫径鈺傜毜閸滃本婀弶銉﹀付閸?- 娑撱儲鐗搁梻顓犲箚鐠囧嫪鍙?- 妫板嫭绁村Ο鈥崇€锋稉搴㈠付閸掑墎绮ㄩ弸婊€绠ｉ梻鏉戞▕瀵倻娈戠化鑽ょ埠閸掑棙鐎?- 婢舵艾褰夐柌蹇氣偓锕€鎮庡鐑樐?
### 娑撳秷鍏橀崡鏇犲娴ｆ粈璐熷鍝勫灡閺傛壆鍋ｉ惃鍕敶鐎?
- 閳ユ粎鏁ゆ禍?Transformer閳?- 閳ユ粌顦块崣鍌涙殶閼帮箑鎮庨垾?- 閳ユ粎鏁ゆ禍?SAC baseline閳?- 閳ユ粌浠涙禍?MPC閳?
鏉╂瑤绨洪崣顏囧厴娴ｆ粈璐熼懗灞炬珯閹存牜绮嶉幋鎰板劥閸掑棴绱濇稉宥堝厴閸楁洜瀚幘鎴ｆ崳鐠佺儤鏋冩稉璇插灡閺傝埇鈧?

## 12. 瑜版挸澧犳导妯哄帥缁?
### 缁楊兛绔存导妯哄帥缁?
閸忓牏菙娴ｅ繑甯堕崚?benchmark閿?
- 瀹告彃鐣幋鎰剁窗`CEMMPC` 閻ㄥ嫬褰叉径宥囧箛閹冩嫲閸╄櫣顢呯粙鍐茬暰閹?- 濮濓絽婀崑姘剧窗妤犲矁鐦?`DLinear / Transformer / Transformer-hybrid` 閸︺劍娲块梹?rollout 娑撳娈戦梻顓犲箚閹烘帒鎮?- 娑撳绔村銉窗闁劖顒為幎?`sp -> actuator feedback -> climate` 閻?surrogate 閺囧瓨鏌婇崑姘杽

缁楊兛绨╃仦鍌滄埛缂侇叀藟瀵椽顣╁ù?benchmark閿?
- 瀹告彃鎯庨崝?`hybrid residual model`
- 娑撳绔村銉︽Ц缂?`hybrid residual model` 鐠烘垶顒滃蹇涱暕缁犳绱濋獮鏈电瑢 `DLinear / Transformer / current hybrid-transformer` 閸嬫氨绮烘稉鈧崣锝呯窞鐎佃鐦?
### 缁楊兛绨╂导妯哄帥缁?
閹?AGC 娑撹崵鍤庨幒銉ュ煂閹貉冨煑鐏炲偊绱?
- 娴犲骸缍嬮崜?surrogate closed-loop 缂佈呯敾閹恒劏绻橀崚鐗堟纯娑撱儲鐗搁惃?AGC 闂傤厾骞嗛悳顖氼暔
- 閸愬秶婀?SAC on AGC

### 缁楊兛绗佹导妯哄帥缁?
閹跺﹨绁┃鎰瘹閺嶅洨鎾奸崗銉窗

- `Heat_cons`
- `ElecHigh`
- `ElecLow`
- `CO2_cons`
- `Irr`

閸?economic MPC 瀵ゆ湹鍑犻妴?

## 13. 瑜版挸澧犲銉ょ稊鐟欏嫬鍨?
1. 閺傛澘绱戦崣鎴滅喘閸忓牊鏂侀崷?[agc_mpc](c:/repositories/strawberry/agc_mpc)閵?2. 闂勩倝娼張澶嬫绾噣娓剁憰渚婄礉娑撳秷顩︾紒褏鐢婚幎濠佸瘜瀹搞儰缍斿ù浣哥垻閸?`diffmpc`閵?3. 娴狅絿鐖滄妯款吇鏉╂劘顢戦悳顖氼暔閺?`strawberry_env`閵?4. 濮ｅ繑顐奸崑姘暚閸忔娊鏁禒锝囩垳閺€鐟板З閵嗕礁鐤勬宀€绮ㄩ弸婊勬纯閺傜増鍨ㄧ捄顖滃殠閸欐ê瀵查崥搴礉闁€燁洣閺囧瓨鏌婇張顒佹瀮娴犺翰鈧?5. 瑜版挸澧犻幒褍鍩楅張顖濐嚔缁撅箑鐣鹃敍?   - `GradientMPC` = 闁俺绻冨顖氬閻╁瓨甯村Ч鍌澬掑姘З閺冭泛鐓欐导妯哄闂傤噣顣介惃?MPC 濮瑰倽袙閸?   - `CEMMPC` = 闁俺绻?CEM 闁插洦鐗遍幖婊呭偍濮瑰倽袙閸氬奔绔?MPC 閻╊喗鐖ｉ惃?MPC 濮瑰倽袙閸?   - 娑撳秴鍟€閹?`DPC` 閸?`MPC` 鐠佺増鍨氭稉銈勯嚋楠炲磭楠囬懠鍐ㄧ础閿涘奔浜掗崗宥嗘钩鐠囶厽璐╁ǎ?6. 娴犺缍嶉弬鐗埬侀崹瀣厴鐟曚礁鎮撻弮璺烘礀缁涙柨娲撴稉顏堟６妫版﹫绱?   - 缁傝崵鍤庢０鍕ゴ閺勵垰鎯侀幓鎰磳
   - 闂傤厾骞嗛幒褍鍩楅弰顖氭儊閹绘劕宕?   - 鐎?forecast error 閺勵垰鎯佺粙鍐蹭淮
   - 閺勵垰鎯侀懗鍊熜掗柌濠佽礋闂堛垹鎮滈幒褍鍩楅惃鍕啎鐠?7. Git 閹绘劒姘︽妯款吇闁插洨鏁ら垾婊冪毈濮濄儱鍨庡▓鍨絹娴溿倐鈧繐绱濇稉宥堫洣閹跺﹦绮ㄩ弸婊呮窗瑜版洟鍣搁弸鍕┾偓浣鼓侀崹瀣煀婢х偑鈧焦甯堕崚璺虹杽妤犲瞼绮ㄩ弸婧库偓浣规瀮濡楋絾娲块弬棰佺濞嗏剝鈧勮穿閹存劒绔存稉顏勩亣閹绘劒姘﹂妴?8. 瑜版挸澧犳禒鎾崇氨閸︺劍婀伴張杞扮瑐閺囨儳鍤悳?`.git` ACL / `index.lock` 閸愭瑥鍙嗛崣妤呮闂傤噣顣介敍娑橆洤閺?`git add` / `git commit` 閹?`Unable to create .git/index.lock: Permission denied`閿?   - 娑撳秷顩﹂崣宥咁槻闁插秷鐦鍫濐樋濞?   - 閸忓牊顥呴弻?`.git` 閻?ACL
   - 韫囧懓顩﹂弮鏈电濞嗏剝鈧団偓鎺戠秺缁夊娅?`.git` 娑撳鎷＄€电懓缍嬮崜宥囨暏閹撮娈?`DENY` ACL 閸氬骸鍟€缂佈呯敾閹绘劒姘?9. 閹恒劏宕橀惃鍕絹娴溿倖濯堕崚鍡涖€庢惔蹇ョ窗
   - 閸忓牊褰佺紒鎾寸亯閻╊喖缍嶇紒鎾寸€?/ plotting / 閸╄櫣顢呯拋鐐煢
   - 閸愬秵褰侀弬鐗埬侀崹瀣╃瑢 forecasting 缂佹挻鐏?   - 閺堚偓閸氬孩褰?control benchmark閵嗕胶绮ㄩ弸婊冩禈 / summary 閸?`CONTEXT.md`
10. 婵″倹鐏夐崥搴ｇ敾 push 閸?pack 鏉╁洤銇囬幋鏍︾癌鏉╂稑鍩楃紒鎾寸亯鏉╁洤顦挎径杈Е閿涘奔绱崗鍫ｂ偓鍐缂佈呯敾閹峰棙褰佹禍銈忕礉韫囧懓顩﹂弮鑸靛Ω閳ユ粈鍞惍浣稿綁閺囩补鈧繂鎷伴垾婊冪杽妤犲奔楠囬悧鈹锯偓婵嗗瀻瀵偓婢跺嫮鎮婇敍宀冣偓灞肩瑝閺勵垱妫ら梽鎰板櫢鐠?push閵?11. 瑜版挸澧犻悳顖氼暔娑撳绱漙Remove-Item` 娑撯偓缁鍨归梽銈呭З娴ｆ粈绡冮崣顖濆厴鐞氼偅鐭欑粻杈ㄥ娴ｅ繐鑻熼幎?`Access is denied`閿涘苯宓嗘担鎸庢瀮娴?ACL 閻鎹ｉ弶銉︻劀鐢潻绱辨俊鍌涚亯闂団偓鐟曚焦绔婚悶?legacy 缂佹挻鐏夐弬鍥︽閿?   - 閸忓牆灏崚鍡樻Ц濞屾瑧顔?閹绘劖娼堥梽鎰煑鏉╂ɑ妲搁弬鍥︽閼奉亣闊?ACL 闂傤噣顣介敍灞肩瑝鐟曚線绮拋銈嗘Ц閺傚洣娆㈤幑鐔锋綎
   - 娴兼ê鍘涢悽銊⑩偓婊呯翱绾喛绻冨?+ 閹绘劖娼堥崚鐘绘珟閳ユ繐绱濇稉宥堫洣閻劋绱扮拠顖欐縺閺傜増鏋冩禒鍓佹畱鐎硅姤纭鹃柅姘跺帳
   - 娓氬顩у〒鍛倞閺冄勫付閸掑墎绮ㄩ弸婊勬閿涘苯褰ч崚鐘绘珟閺冄冩嚒閸氬秶娈?`_dpc_` 閸滃本妫?`_mpc_` 閺傚洣娆㈤敍灞肩瑝鐟曚礁灏柊宥呭煂 `gradient_mpc` / `cem_mpc`


## 14. 娑撳顐肩€电鐦藉楦款唴鐠ч攱澧滈崘鍛啇

瀵ら缚顔呴崗鍫ｎ嚛閺勫函绱?
- 瑜版挸澧犳稉濠氥€嶉惄顔炬窗瑜版洩绱癭agc_mpc`
- 瑜版挸澧犳稉缁樻殶閹诡噣娉﹂敍姝欰utonomousGreenhouseChallenge_edition2`
- 瑜版挸澧犲鎻掔暚閹存劧绱伴弫鐗堝祦缁狅紕鍤?+ GRU baseline + DLinear baseline
- 瑜版挸澧犲鎻掔暚閹存劧绱伴弫鐗堝祦缁狅紕鍤?+ GRU baseline + DLinear baseline + SegRNN baseline + Transformer baseline + Transformer-hybrid baseline + hybrid residual 閸樼喎鐎?+ 閼奉亜濮╃紒鎾寸亯閸?- 瑜版挸澧犲鎻掔暚閹存劧绱癭DLinear / Transformer / Transformer-hybrid` 瀹稿弶甯撮崗?AGC 娑撳﹦娈?`GradientMPC / CEMMPC` 閸掓繄澧?surrogate closed-loop benchmark
- 瑜版挸澧犲鎻掔暚閹存劧绱癴orecast 娓氀勬煀婢?rolling multi-step rollout 閸ユ拝绱眂ontrol 娓氀囩帛鐠併倕鍨忛崚鐗堟纯娑撱儲鐗搁惃?`surrogate` rollout閿涘苯鑻熸宀冪槈娴?`CEMMPC` 閻ㄥ嫬褰叉径宥囧箛閹?- 瑜版挸澧犳稉瀣╃濮濄儻绱伴幒褍鍩楁笟褏鎴风紒顓熷Ω surrogate 娴?`state persistence + action proxy` 閹恒劌鍩岄弴瀛樺复鏉?`sp -> vip -> actuator -> climate` 閻ㄥ嫬鐪扮痪褍缂撳Ο鈽呯幢妫板嫭绁存笟褎濡?`hybrid residual model` 鐠烘垶鍨氬锝呯础妫板嫮鐣婚獮璺轰粵缂佺喍绔寸€佃鐦?## 15. Strawberry vs AGC 鐎佃鐦崶?
- 瀹稿弶鏌婃晶鐐差嚤鐢牆鐫嶇粈铏规暏閼存碍婀伴敍姝攃ompare_dataset_switch.py](c:/repositories/strawberry/agc_mpc/compare_dataset_switch.py)
- 鏉╂劘顢戦弬鐟扮础閿?  ```bash
  conda activate strawberry_env
  python c:\repositories\strawberry\agc_mpc\compare_dataset_switch.py
  ```
- 鏉堟挸鍤弬鍥︽閿?  - [strawberry_vs_agc_dataset_switch.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/strawberry_vs_agc_dataset_switch.png)
  - [strawberry_vs_agc_dataset_switch_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/strawberry_vs_agc_dataset_switch_summary.json)
- 閸ュ墽娈戝В鏃囩窛閸欙絽绶為敍?  - 閸欘亝鐦潏鍐ㄥ彙閸氬苯褰夐柌?`Temperature / Humidity / CO2`
  - 閸欘亝鐦潏?final-step 閹稿洦鐖?  - 娑撱倛绔熼柈鑺ュ瘻閳? 鐏忓繑妞傛０鍕ゴ娴犺濮熼垾婵嗩嚠姒绘劧绱伴弮?Strawberry = `120 x 1 min`閿涘瓑GC = `24 x 5 min`
- 瑜版挸澧犵紒鎾诡啈閿?  - 閺?Strawberry Transformer-hybrid 閻?final MAE 娑?`3.36 / 6.78 / 105.88`
  - AGC `DLinear` 閻?final MAE 娑?`0.76 / 4.46 / 54.73`
  - AGC `Transformer` 閻?final MAE 娑?`0.82 / 4.92 / 47.23`
  - AGC `Transformer-hybrid` 閻?final MAE 娑?`0.77 / 5.31 / 58.32`
  - 閺?Strawberry 閸?`CO2` 娑?final `R2` 閸欘亝婀?`0.073`閿涙睔GC 娑撳閲滃Ο鈥崇€风€电懓绨叉稉?`0.776 / 0.824 / 0.743`
- 鐎电懓顕辩敮鍫㈡畱閹恒劏宕樼悰銊ㄥ牚閿?  - 鏉╂瑥绱堕崶鍙ョ瑝鐠囦焦妲?閳ユ穾GC 瀹歌尙绮￠崑姘煂閻炲棙鍏傞弸渚€妾洪垾?  - 鐎瑰啳鐦夐弰搴ｆ畱閺勵垽绱伴崷銊ョ秼閸?baseline-first 鐎圭偟骞囨稉瀣剁礉AGC 瀹歌尙绮￠懗鑺ュ絹娓氭稒娲跨粙鍐茬暰閵嗕焦娲块崣顖涘付閵嗕礁顕梻顓犲箚閺囨潙寮告總鐣屾畱妫板嫭绁撮崺鍝勯獓
- 閸ョ姵顒濋崚鍥ㄥ床閺佺増宓侀梿鍡欐畱娑撴槒顩﹂悶鍡欐暠鎼存棁銆冩潻棰佽礋閳ユ粈鎹㈤崝鈥冲爱闁板秴瀹抽弴鎾彯 + 缂佹挻鐏夐弴瀵盖?+ 閼冲€熷殰閻掕埖澧跨仦鏇炲煂闂傤厾骞嗛幒褍鍩楅垾婵撶礉閼板奔绗夐崣顏呮Ц閳ユ粍妫弫鐗堝祦闂嗗棗鍨庨弫鏉挎▕閳?- 瀹稿弶鏌婃晶鐐板敩鐞涖劍鈧囶暕濞村鐛ョ€佃鐦崶鎾呯窗[strawberry_vs_agc_forecast_windows.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/strawberry_vs_agc_forecast_windows.png)
- 鐠囥儱娴橀崣顏勭潔缁€?`Strawberry / old Transformer-hybrid`閵嗕梗AGC / Transformer`閵嗕梗AGC / Transformer-hybrid`
- 鐠囥儱娴樻担璺ㄦ暏娑撱倛绔熷ù瀣槸闂嗗棗鎮囬懛顏嗘畱 midpoint sample閿涘奔绗夐崑姘壉閺堫剙顕鎰剁礉娑撳秶鏁ゆ禍搴濆紬閺嶈偐绮虹拋鈩冪槷鏉堝喛绱濋崣顏嗘暏娴滃海绮扮€电厧绗€閸嬫埃鈧粓顣╁ù瀣缓鏉╃懓鑸伴幀浣测偓婵堟畱閻╃顫囩拠瀛樻
- forecasting 閸ユ儳鍑￠崡鍥╅獓娑撹　鈧粌娴橀崘鍛纯閹恒儲妯夌粈鐑樺瘹閺嶅洠鈧繐绱?- `forecast_examples / rollout / horizon_mae` 閻滄澘婀柈鎴掔窗閻╁瓨甯撮崗瀹犱粓瑜版挸澧犲Ο鈥崇€烽惃?`Full R2 / Full MAE / Final R2 / Final MAE`
  - `horizon_mae` 閸ュ彞绱伴崷銊ユ禈娑撳鏌熷Ч鍥ㄢ偓璇插弿闁劎娲伴弽鍥╂畱閹稿洦鐖?- control 閸ユ儳鍑￠崡鍥╅獓娑撹　鈧粎濮搁幀?+ 閹稿洦鐖?+ 閸斻劋缍旈垾婵婁粓閸氬牆鐫嶇粈鐚寸窗
  - 閸撳秴娲撶悰灞肩矝閺?`Tair / Rhair / CO2air / Tot_PAR`
  - 缁楊兛绨茬悰灞炬▔缁€?`objective / |u-u_log| / action_tv`
  - 缁楊剙鍙氱悰灞炬▔缁€鐑樺閺堝甯堕崚鍫曞櫤閻ㄥ嫬缍婃稉鈧崠鏍уЗ娴ｆ粏寤烘潻鐧哥礉鐎圭偟鍤庨弰?executed閿涘矁娅勭痪鎸庢Ц logged baseline
- 瀹稿弶鏌婃晶鐐存瀮閻氼喖顕悡褎鏋冨锝忕窗[LITERATURE_COMPARISON.md](c:/repositories/strawberry/agc_mpc/LITERATURE_COMPARISON.md)
- 閺傚洨灏炵€靛湱鍙庨弬鍥ㄣ€傞惃鍕暰娴ｅ稄绱?  - 娑撳秴浠涙导?leaderboard
  - 閹稿鎹㈤崝掳鈧浇绶崗銉ｂ偓浣界翻閸戞亽鈧弓orizon閵嗕焦膩閸ㄥ鈧焦甯堕崚鎯邦啎鐎规哎鈧胶绮ㄩ弸婊冩嫲閸欘垱鐦幀褍鍨庡鈧崘?  - 瑜版挸澧犵紒鎾诡啈閺勵垽绱癆GC 缂佹挻鐏夋潻妯圭瑝閺?final-paper quality閿涘奔绲惧鎻掝槱娴滃骸褰叉潏鈺傚Б閻?literature band 閸愬拑绱遍惇鐔割劀閻厽婢橀崷?`Rhair`閵嗕菇ncertainty閵嗕躬conomic objective 閸滃本娲跨€瑰本鏆ｉ梻顓犲箚
- 瀹稿弶鏌婃晶鐐躲€冮弽鐓庣础鏉╂垶婀＄拋鐑樻瀮缂佽壈鍫弬鍥ㄣ€傞敍姝擱ECENT_PAPERS_SURVEY.md](c:/repositories/strawberry/agc_mpc/RECENT_PAPERS_SURVEY.md)
- 鐠囥儲鏋冨锝嗗瘻閳ユ粏顔戦弬?/ 娴犺濮?/ 娑撶粯膩閸?/ 鐎佃鐦?baseline / 閸氼垰褰?/ 闁剧偓甯撮垾婵堢矋缂佸浄绱濋崚鍡曡礋閿?  - 濞撯晛顓绘０鍕ゴ鐠佺儤鏋?  - 濞撯晛顓婚幒褍鍩楃拋鐑樻瀮
  - 闁氨鏁ら弮璺虹碍濡€崇€烽崣鍌濃偓?- 閻劑鈧棑绱?  - 韫囶偊鈧喎娲栫粵鏂衡偓婊勬付鏉╂垹娴夋导鑹邦啈閺傚洭鍏橀悽銊ょ啊娴犫偓娑斿牊膩閸ㄥ鈧攻aseline 閸滃苯顕В鏂款嚠鐠炩剝妲告禒鈧稊鍫氣偓?  - 娑撳搫鎮楃紒顓熸煀濡€崇€风捄顖滃殠閹绘劒绶甸弬鍥╁盀闁挎氨鍋ｉ敍宀勪缉閸忓秴寮芥径宥嗗瀹搞儲鏆ｉ悶?- 瀹告彃婀拠銉︽瀮濡楋絼鑵戠悰銉ュ帠 `Mao et al., 2024` 閻ㄥ嫰鍣搁悙纭咁嚊鐟欙絽鐨懞鍌︾礉娑撴捇妫崶鐐电摕閿?  - 娑撹桨绮堟稊鍫ｎ嚉閺?`PSO-BiGRU-Attention-LightGBM` 閻?`R2` 瀵板牓鐝?  - 鐎瑰啫鎷拌ぐ鎾冲 `AGC` 閺佺増宓侀梿鍡楀煂鎼存洘婀佹径姘辨祲娴?  - 鐎瑰啯妲搁崥锕€褰叉禒銉潶娑撱儲鐗告径宥囧箛閿涘苯鎽㈡禍娑㈠劥閸掑棗褰ч懗钘変粵閺傝纭剁痪褍顦查悳?- 鐠囥儲鏋冨锝囧箛瀹稿弶瀵滆ぐ鎾冲娑撹崵鍤庣悰銉ュ繁楠炲爼鍣搁崘娆庤礋楠炴彃鍣?UTF-8 閻楀牊婀伴敍灞炬煀婢х偞鍨ㄥ鍝勫娴滃棴绱?  - `Zeng et al., 2022 / DLinear`
  - `PatchTST`
  - `iTransformer`
  - `TimeMixer`
  - `SAMformer`
  - `ETSformer`
  - `FreTS`
  - `OneNet`
- 瑜版挸澧犻弴瀛樻绾喚娈戦弬鍥╁盀缂佹捁顔戦弰顖ょ窗
  - 娑撳秴鈧厧绶辩紒褏鐢婚崼?plain Transformer
  - 閺囨潙鎮庨悶鍡欐畱閺?`DLinear main path + stronger residual branch`
  - 瑜版挸澧犻張鈧崐鐓庣繁娴兼ê鍘涚捄鎴炵濡ゆ氨娈戞稉澶嬫蒋 residual 鐠侯垳鍤庨弰?`Transformer-hybrid / iTransformer / PatchTST residual`
- `README.md` 瀹歌尪藟閸忓懏鏆熼幑顕€娉﹂懗灞炬珯娑撳氦顔勭紒鍐啎鐎规俺顕╅弰搴窗
  - 閺勫海鈥?`AGC` 閺囨潙鍣涵顔芥Ц multi-compartment benchmark閿涘矁鈧奔绗夐弰?fully independent multi-greenhouse dataset
  - 鐞涖儱鍘栬ぐ鎾冲 `x_past / w_future / u_future / y_future` 閻ㄥ嫭甯堕崚璺侯嚤閸氭垶甯撮崣锝堫嚛閺?  - 鐞涖儱鍘?single-compartment training 娑?multi-compartment joint training 閻ㄥ嫬褰囬懜宥忕礉瑜版挸澧犳妯款吇娴犲秳浜?joint training 娑撹桨瀵?- 瀹稿弶鏌婃晶鐐额唲缂佸啰鐡ラ悾銉ヮ嚠閻撗嗗壖閺堫剨绱癧compare_training_regimes.py](c:/repositories/strawberry/agc_mpc/compare_training_regimes.py)
- 鐠囥儴鍓奸張顒佹暜閹镐礁娲跨紒鏇氱娑擃亞娲伴弽鍥闂傚瓨鐦潏鍐х瑏缁?regime閿?  - `single`: 閸欘亜婀惄顔界垼闂呮棃妫挎稉濠咁唲缂佸啫鑻熼崷銊嚉闂呮棃妫垮ù瀣槸
  - `joint_all`: 閸︺劌鍙忛柈銊╂闂傜繝绗傜拋顓犵矊閿涘奔绲鹃崣顏勬躬閻╊喗鐖ｉ梾鏃堟？濞村鐦?  - `leave_one_out`: 閸︺劑娅庨惄顔界垼闂呮棃妫挎径鏍畱閸忔湹缍戦梾鏃堟？娑撳﹨顔勭紒鍐跨礉閸愬秴婀惄顔界垼闂呮棃妫垮ù瀣槸
- 閺佺増宓佺粻锛勫殠瀹稿弶鏌婃晶鐐跺殰鐎规矮绠?bundle 缂佸嫯顥婇懗钘夊閿涘苯褰查幐?train/eval compartments 閼奉亞鏁遍幏鍏煎复楠炴湹绮庨悽銊唲缂佸啴娉﹂幏鐔锋値 scaler
- 缂佹挻鐏夌紒鐔剁閽€钘夊煂閿涙瓪agc_mpc/results/forecasting/analysis`
- 瀹告彃浠?1-epoch smoke test閿涘牏娲伴弽鍥闂?`Reference`閿涘本膩閸?`DLinear`閿涘绱?  - `single`閿涙瓪Tair/Rhair/CO2air/Tot_PAR` Final MAE = `0.772 / 4.815 / 93.219 / 53.889`
  - `joint_all`閿涙瓪0.776 / 3.798 / 53.866 / 32.658`
  - `leave_one_out`閿涙瓪0.671 / 5.469 / 56.336 / 38.663`
- 閸掓繃顒炴穱鈥冲娇閿?  - joint training 鐎?`Rhair / CO2air / Tot_PAR` 閺勫孩妯夐弴瀛樻箒鐢喖濮?  - leave-one-out 閸?`Reference` 閻?`Tair` 娑撳﹤绶㈠鐚寸礉娴ｅ棗顕﹢鍨閸?CO2 娑撳秴宕版导?  - 閸楁洟娈ч梻纾嬵唲缂佸啫鑻熸稉宥呫亯閻掕埖娲挎總鏂ょ礉閼峰啿鐨崷銊ョ秼閸?`Reference + DLinear` 閻?smoke test 娑撳﹣绗夐弰?- 瀹稿弶鏌婃晶?`diffmpc` 妞嬪孩鐗?Transformer 鏉╀胶些閸╁搫鍣懘姘拱閿涙瓟benchmark_diffmpc_style_transformer.py](c:/repositories/strawberry/agc_mpc/benchmark_diffmpc_style_transformer.py)
- 鐠囥儴鍓奸張顒傛畱閻╊喚娈戞稉宥嗘Ц鏉╄棄缍嬮崜?`agc_mpc` 閺堚偓瀵搫鍨庨弫甯礉閼板本妲搁幒褍鍩楅崣姗€鍣洪崷鏉挎礀缁涙棑绱?  - 閸︺劌鏁栭柌蹇庣箽閻ｆ瑦妫?`diffmpc` Transformer-hybrid 閺嬭埖鐎稉搴ゎ唲缂佸啴顣╃粻妤佹閿涘畭AGC` 閺勵垰鎯佸В鏃€妫?Strawberry 閺囨挳鈧倸鎮庢担婊€璐?Transformer 閻ㄥ嫭鏆熼幑顔肩唨鎼?- 閸ュ搫鐣鹃崡蹇氼唴閿?  - targets = `Tair / Rhair / CO2air`
  - `seq_len = 48`閿涘牆顕惔鏃€妫い鍦窗 `240 min` 閸樺棗褰堕敍?  - `horizon = 24`閿涘牆顕惔鏃€妫い鍦窗 `120 min` 妫板嫭绁寸粣妤嬬礆
  - `d_model = 64`, `nhead = 4`, `num_layers = 4`, `ff_dim = 128`, `dropout = 0.1`
  - `batch_size = 256`, `num_epochs = 200`, `lr = 1e-4`, `lambda_trend = 0.3`, `patience = 15`
- 鐠佹崘顓搁崢鐔峰灟閿?  - 姒涙顓婚崣顏囨儰 summary JSON閿涘奔绗夐懛顏勫З閻㈢喐鍨氭径褍娴?  - 閸忓牊濡搁垾婊勀侀崹瀣波閺?鐠侇厾绮屾０鍕暬/閺冨爼妫块崣锝呯窞閳ユ繂顕鎰剁礉閸愬秷鐨ラ弫鐗堝祦闂嗗棙妲搁崥锔芥纯闁倸鎮?Transformer
- 瀹告彃浠?1-epoch smoke test閿涘潉single + Reference`閿涘鑻熼幋鎰閽€鐣屾磸閿?  - [diffmpc_style_transformer_single_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_single_reference_summary.json)
  - 瑜版挸澧犳禒鍛暏娴滃酣鐛欑拠浣稿弳閸欙絼绗岄崡蹇氼唴閿涘奔绗夐悽銊ょ艾濮濓絽绱＄紒鎾诡啈
- 鐠囥儱鐔€閸戝棛骞囧鎻掔暚閹?`Reference` 娑撳﹦娈戝锝呯础娑撳绮嶆潻鎰攽閿?  - [diffmpc_style_transformer_single_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_single_reference_summary.json)
  - [diffmpc_style_transformer_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_joint_all_reference_summary.json)
  - [diffmpc_style_transformer_leave_one_out_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/diffmpc_style_transformer_leave_one_out_reference_summary.json)
- `diffmpc` 妞嬪孩鐗?Transformer 閸?AGC / `Reference` 娑撳﹦娈戦張鈧紒鍫熷瘹閺嶅浄绱?  - `single`
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
- 瑜版挸澧犵拠缁樼《閿?  - 閺?`diffmpc` 妞嬪孩鐗哥紒鎾寸€潻浣稿煂 AGC 閸氬函绱漙Tair / CO2air` 閺勫孩妯夋總鎴掔艾閺?Strawberry 娑撳﹦娈戦弮?Transformer-hybrid 缂佹挻鐏夐敍宀冾嚛閺勫孩鏆熼幑顕€娉﹂崚鍥ㄥ床绾喖鐤勭敮顔煎И娴滃棜绻栫猾?conditional Transformer
  - 娴?`Rhair` 濞屸剝婀侀崥灞绢劄閸欐ɑ鍨氬娲€嶉敍宀冾嚛閺勫簶鈧粍鏆熼幑顕€娉﹂弴鎾偓鍌氭値 Transformer閳ユ繀绗夌粵澶夌艾閳ユ粍妫紒鎾寸€弮鐘绘付閺€褰掆偓鐘叉皑娴兼艾鍙忛棃銏犲綁瀵　鈧?  - 娑撳顫?AGC 鐠侇厾绮?regime 濞屸剝婀侀崡鏇氱缂佹繂顕張鈧导姗堢窗
    - `single` 閸?`Rhair` 娑撳﹥娓舵總?    - `joint_all` 閸?`CO2air` 娑撳﹥娓舵總?    - `leave_one_out` 閸?`Tair` 娑撳﹥娓舵總?  - 閸ョ姵顒濈€电懓顕辩敮鍫熸纯缁嬪磭娈戠悰銊ㄥ牚鎼存梹妲搁敍?    - AGC 缂佹瑦妫?Transformer 妞嬪孩鐗搁幓鎰返娴滃棙娲块崥鍫㈡倞閻ㄥ嫭鏆熼幑顔藉复閸欙絽鎷伴弴鎾彯閻ㄥ嫪绗傞梽鎰敄闂?    - 娴ｅ棛婀″锝嗗Ω鐠囥儲鐏﹂弸鍕粵瀵尨绱濇禒宥囧姧闂団偓鐟曚浇绻樻稉鈧銉╂桨閸?AGC/閹貉冨煑娴犺濮熼弨褰掆偓鐙呯礉閼板奔绗夐弰顖滄纯閹恒儳鍙庨幖顒佹＋缂佹挻鐎?- 瀹稿弶鏌婃晶鐐垫纯鐟欏倸顕В鏂挎禈閿涙瓟diffmpc_style_transformer_dataset_suitability.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/diffmpc_style_transformer_dataset_suitability.png)
- 鐠囥儱娴橀崣顏呯槷鏉堝喛绱?  - `Strawberry / old Transformer-hybrid`
  - `AGC / diffmpc-style / single`
  - `AGC / diffmpc-style / joint_all`
  - `AGC / diffmpc-style / leave_one_out`
- 鐠囥儱娴橀惃鍕暰娴ｅ稄绱?  - 閻劋绨惄纾嬵潎鐏炴洜銇氶垾婊冩晼闁插繒娴夋导鑲╂畱 Transformer 妞嬪孩鐗告稉搴ゎ唲缂佸啴顣╃粻妞烩偓婵呯瑓閿涘本宕查崚?AGC 閸?`Temperature / Humidity / CO2` 閻?final MAE 娑?final R2 婵″倷缍嶉崣妯哄
  - 娑撳秵璐╅崗銉ョ秼閸?`agc_mpc` 閻?`DLinear / Transformer / Transformer-hybrid` 閺?baseline閿涘矂浼╅崗宥堫啈鐠囦礁褰涘鍕磽缁?- 瀹稿弶鏌婃晶鐐存纯闁倸鎮庡Ч鍥ㄥГ閻ㄥ嫪琚卞鐘辫厬閺傚洤娴橀敍?  - [diffmpc_style_transformer_best_vs_old_line_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/diffmpc_style_transformer_best_vs_old_line_cn.png)
  - [diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/diffmpc_style_transformer_old_vs_agc_joint_all_windows_cn.png)
- 娑撱倕娴樼€规矮缍呴敍?  - `best_vs_old_line_cn`閿涙艾褰ч惇?`old Strawberry` vs `AGC joint_all`閿涘瞼鏁ゆ稉顓熸瀮閹舵鍤庨崶鎯х潔缁€?`Temperature / Humidity / CO2` 閻?final MAE 娑?final R2
  - `old_vs_agc_joint_all_windows_cn`閿涙艾鑻熼幒鎺戠潔缁€鐑樻＋ Strawberry 娑?AGC joint_all 閻ㄥ嫪鍞悰銊︹偓褔顣╁ù瀣崶閿涘矁顔€鐎电厧绗€閻╁瓨甯撮惇瀣缓鏉╃鍒涢崥鍫滅瑢閸嬪繒些閺傜懓绱?- 瀹稿弶鏌婃晶鐐┾偓婊勬＋閺佺増宓侀梿鍡樻＋ hybrid-transformer vs 閺傜増鏆熼幑顕€娉﹂弬?hybrid-transformer閳ユ繄娈戦崗顒€閽╂０鍕暬鐎靛湱鍙庢稉鑽ゅ殠閿?  - 閺冄傛櫠閿涙瓪diffmpc` 閸樼喎顫?`TransformerHybridModel`
  - 閺傞鏅堕敍姝歛gc_mpc` 瑜版挸澧?`ConditionalTransformerHybridForecaster`
  - 閸忓崬鎮撻崣锝呯窞閿涙艾褰ч惇?`Tair / Rhair / CO2air`閿涘瞼绮烘稉鈧幐?`2h` 妫板嫭绁存禒璇插鐠併劏顔?  - 閺冄傛櫠娣囨繄鏆€閺冄囥€嶉惄顔芥煙濞夋洑绗岄弸鑸电€敍娑欐煀娓氀傜箽閻?AGC 瑜版挸澧?`x_past / w_future / u_future -> y_future` 閻?control-oriented 閹恒儱褰?- 瀹稿弶鏌婃晶鐐跺壖閺堫剨绱癧benchmark_current_hybrid_transformer.py](c:/repositories/strawberry/agc_mpc/benchmark_current_hybrid_transformer.py)
  - 閻╊喚娈戦敍姘辩舶 AGC 瑜版挸澧?hybrid-transformer 娑撯偓娑擃亝鐦?12 epoch baseline 閺囨潙鍙曢獮宕囨畱鐠侇厾绮屾０鍕暬閿涘苯鍟€娑撳孩妫?Strawberry 閻?old hybrid-transformer 閸嬫艾顕В?  - 瑜版挸澧犲锝呯础鐠烘垿鈧氨娈戦柊宥囩枂娑撶尨绱癭joint_all + Reference`
  - 鐠侇厾绮屾０鍕暬閿涙瓪batch_size=256`, `num_epochs=200`, `lr=1e-4`, `lambda_trend=0.3`, `patience=15`
  - 濡€崇€烽崣鍌涙殶閿涙瓪hidden_dim=96`, `nhead=4`, `num_layers=2`, `ff_dim=192`, `dropout=0.1`
- 瑜版挸澧犲锝呯础缂佹挻鐏夐弬鍥︽閿?  - [current_hybrid_transformer_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/current_hybrid_transformer_joint_all_reference_summary.json)
  - [current_hybrid_transformer_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/current_hybrid_transformer_joint_all_reference.pt)
- `AGC + current hybrid-transformer + joint_all + Reference` 濮濓絽绱＄紒鎾寸亯閿?  - `Tair`: Full `R2=0.9344`, MAE `0.630`; Final `R2=0.9318`, MAE `0.651`
  - `Rhair`: Full `R2=0.8951`, MAE `3.698`; Final `R2=0.8553`, MAE `4.403`
  - `CO2air`: Full `R2=0.8184`, MAE `41.201`; Final `R2=0.7860`, MAE `44.567`
- 娑?`AGC + diffmpc-style hybrid-transformer + joint_all + Reference` 閻ㄥ嫮娲块幒銉ヮ嚠濮ｆ棑绱?  - `Tair`: Final MAE `1.358 -> 0.651`, Final `R2 0.8007 -> 0.9318`
  - `Rhair`: Final MAE `7.891 -> 4.403`, Final `R2 0.6470 -> 0.8553`
  - `CO2air`: Final MAE `72.867 -> 44.567`, Final `R2 0.3899 -> 0.7860`
- 瑜版挸澧犻弴瀵盖旈惃鍕€冩潻鏉跨安閺€閫涜礋閿?  - 娑撳秵妲搁垾娣嶨C 閼奉亜濮╃拋鈺傛＋ Transformer 閸欐ê宸遍垾?  - 閼板本妲搁垾娣嶨C 閺囨挳鈧倸鎮庤ぐ鎾冲鏉╂瑥顨滈棃銏犳倻閹貉冨煑閻?hybrid-transformer 閹恒儱褰涙稉搴ゎ唲缂佸啳瀵栧蹇娾偓?  - 閺?Strawberry + old hybrid-transformer 娑?AGC + current hybrid-transformer 閻ㄥ嫬顕В鏃撶礉閹靛秵娲块懗鑺ユ暜閹镐讲鈧ɑ宕查弫鐗堝祦闂?+ 閹广垺鏌熷▔鏇熸Ц閸氬牏鎮婃稉鑽ゅ殠閳ユ瑨绻栨稉鈧紒鎾诡啈
- 瀹稿弶鏌婃晶?`AGC + current hybrid-transformer + joint_all + Reference + horizon=120` 濮濓絽绱＄€圭偤鐛欓敍?  - [current_hybrid_transformer_h120_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/current_hybrid_transformer_h120_joint_all_reference_summary.json)
  - [current_hybrid_transformer_h120_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/current_hybrid_transformer_h120_joint_all_reference.pt)
  - 濞夈劍鍓伴敍姘崇箹闁插瞼娈?`120-step` 閹?`120 x 5min = 600 min`閿涘奔绗夐崘宥囩搼娴犺渹绨弮?Strawberry 閻?`120 x 1min = 120 min`
- `AGC current hybrid-transformer` 閸?`horizon=120` 娑撳娈戝锝呯础缂佹挻鐏夐敍?  - `Tair`: Full `R2=0.9204`, MAE `0.764`; Final `R2=0.9153`, MAE `0.820`
  - `Rhair`: Full `R2=0.7302`, MAE `6.705`; Final `R2=0.7149`, MAE `6.875`
  - `CO2air`: Full `R2=0.5754`, MAE `63.666`; Final `R2=0.5573`, MAE `65.198`
- 娑撳骸缍嬮崜?`horizon=24` 鐎佃鐦惃鍕嚢濞夋洩绱?  - `Tair`: Final MAE `0.651 -> 0.820`
  - `Rhair`: Final MAE `4.403 -> 6.875`
  - `CO2air`: Final MAE `44.567 -> 65.198`
  - 鐠囧瓨妲戦敍姘Ω AGC 娴犺濮熸禒?`2h` 閹峰鍩?`10h` 閸氬函绱濋幀褑鍏橀弰搴㈡▔娑撳妾烽敍灞肩稻 `Tair` 娴犲秳绻氶幐浣界窛瀵尨绱盽Rhair / CO2air` 閺囨潙顔愰弰鎾绘 horizon 閹峰鏆遍懓宀勨偓鈧崠?- 瀹稿弶鏌婃晶鐐拌⒈瀵姳鑵戦弬?horizon 鐎佃鐦崶鎾呯窗
  - [current_hybrid_transformer_h24_vs_h120_metrics_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h24_vs_h120_metrics_cn.png)
  - [current_hybrid_transformer_h24_vs_h120_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h24_vs_h120_windows_cn.png)
  - 閻劑鈧棑绱伴幎?`24-step (120 min)` 閸?`120-step (600 min)` 閺€鎯ф躬閸氬奔绔存い鍏哥瑐閿涘瞼婀呴幐鍥ㄧ垼閸滃矁寤烘潻鐟邦洤娴ｆ洟娈?horizon 閹峰鏆遍懓宀勨偓鈧崠?- 瀹稿弶鏌婃晶鐐存纯缁楋箑鎮庤ぐ鎾冲娑撹崵鍤庨惃鍕⒈瀵姳鑵戦弬鍥ㄧ湽閹躲儱娴橀敍?  - [current_hybrid_transformer_best_vs_old_line_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_best_vs_old_line_cn.png)
  - [current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_old_vs_agc_joint_all_windows_cn.png)
- 瀹稿弶鏌婃晶鐐┾偓婊冨瀻闁界喎顕鎰潔缁€铏瑰閳ユ繄鐛ラ崣锝呮禈閿?  - [current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_old_vs_agc_joint_all_windows_minute_aligned_cn.png)
  - 閻劑鈧棑绱扮拋鈺佷箯娓?`120 x 1min` 閸滃苯褰告笟?`24 x 5min` 閸︺劏顫嬬憴澶夌瑐闁棄鐫嶅鈧崚?`120 min` 閺冨爼妫挎潪杈剧礉娓氬じ绨€电厧绗€閼插婧傚В鏃囩窛
  - 闁插秷顩︾拠瀛樻閿涙艾褰告笟褍褰ч弰顖涘Ω `24 x 5min` 閻ㄥ嫮婀＄€?妫板嫭绁存潪銊ㄦ姉閹绘帒鈧厧鍩?`120` 娑擃亜鍨庨柦鐔哄仯閸嬫碍妯夌粈鐚寸礉娑撳秳鍞悰銊δ侀崹瀣埂閻ㄥ嫬浠涙禍?`120` 濮?AGC 妫板嫭绁?- 鏉╂瑤琚卞鐘叉禈閻ㄥ嫬褰涘鍕Ц閿?  - 瀹革缚鏅堕崶鍝勭暰娑撹　鈧粍妫?Strawberry + old hybrid-transformer閳?  - 閸欏厖鏅堕崶鍝勭暰娑撹　鈧穾GC + current hybrid-transformer + joint_all閳?  - 閻劋绨崥鎴濐嚤鐢牐顕╅弰搴窗閻喐顒滈崐鐓庣繁鐠佽尙娈戞稉宥嗘Ц閳ユ粍妫紒鎾寸€潻浣稿煂閺傜増鏆熼幑顕€娉﹂垾婵撶礉閼板本妲搁垾婊勬煀閺佺増宓侀梿鍡氼唨閺傛壆娈?control-oriented hybrid-transformer 閸欐ê绶遍崥鍫㈡倞娑撴梹婀侀弫鍫氣偓?- 瀹稿弶鏌婃晶鐐磋穿閸氬牊鐣顔侥侀崹瀣壖閺堫剨绱癧benchmark_hybrid_residual_forecaster.py](c:/repositories/strawberry/agc_mpc/benchmark_hybrid_residual_forecaster.py)
  - 鐎规矮缍呴敍姘稊娑撳搫缍嬮崜宥夘暕濞村瀵岀痪璺ㄦ畱娑撳绔村銉礉娑撳秴鍟€缂佈呯敾閸?plain Transformer閿涘矁鈧本妲搁幎?`DLinear` 閻ㄥ嫮菙閸嬨儰瀵岀捄顖氱窞娑?`Transformer-hybrid` 閻ㄥ嫰鐝梼鑸电暙瀹割喖缂撳Ο锛勭波閸氬牐鎹ｉ弶?  - 缂佹挻鐎敍姝欳onditionalDLinearForecaster` 鐠愮喕鐭?main path閿涘畭ConditionalTransformerHybridForecaster` 鐠愮喕鐭?residual path閿涘本娓剁紒鍫ｇ翻閸戣桨璐?`base + gated residual`
  - 瑜版挸澧犲鍙夊复閸?[main.py](c:/repositories/strawberry/agc_mpc/main.py) 閻?baseline 閸忋儱褰涢敍灞肩瘍閺€顖涘瘮閻欘剛鐝?fair-budget benchmark
- 瀹稿弶鏌婃晶鐐拌⒈娑擃亜鎮撻崣锝呯窞 residual 閸婃瑩鈧绱?  - `DLinear + iTransformer residual`
  - `DLinear + PatchTST residual`
- 瀹稿弶鏌婃晶鐐电埠娑撯偓鐎佃鐦懘姘拱閿涙瓟benchmark_residual_forecaster_variants.py](c:/repositories/strawberry/agc_mpc/benchmark_residual_forecaster_variants.py)
  - 瑜版挸澧犳稉澶嬫蒋閺堚偓娴兼ê鍘涙０鍕ゴ闁鐎风紒鐔剁娑撶尨绱?    - `transformer_hybrid_residual`
    - `itransformer_residual`
    - `patchtst_residual`
  - 閻╊喚娈戦敍姘帥閸︺劌鎮撴稉鈧?`fair-budget` 閸楀繗顔呮稉瀣Ω娑撳娼?residual 鐠侯垳鍤庨弨鎯у煂閸氬奔绔撮崣锝呯窞濮ｆ棁绶濋敍灞藉晙閸愬啿鐣剧拫浣界箻閸忋儲甯堕崚鏈垫櫠
- 瀹告彃浠?1-epoch smoke test閿涘潉joint_all + Reference`閿涘argets = `Tair / Rhair / CO2air`閿涘绱?  - [hybrid_residual_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/hybrid_residual_forecaster_joint_all_reference_summary.json)
  - [hybrid_residual_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/hybrid_residual_forecaster_joint_all_reference.pt)
  - `Tair`: Full `R2=0.8960`, MAE `0.912`; Final `R2=0.8904`, MAE `0.925`
  - `Rhair`: Full `R2=0.8828`, MAE `4.145`; Final `R2=0.8435`, MAE `4.887`
  - `CO2air`: Full `R2=0.6480`, MAE `58.135`; Final `R2=0.5861`, MAE `62.729`
- 瀹歌尪藟閸嬫艾鎮撻崡蹇氼唴 `DLinear` 1-epoch quick benchmark閿涘潉joint_all + Reference`閿涘argets = `Tair / Rhair / CO2air`閿涘绱?  - [dlinear_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/dlinear_forecaster_joint_all_reference_summary.json)
  - [dlinear_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/dlinear_forecaster_joint_all_reference.pt)
  - `Tair`: Full `R2=0.8870`, MAE `1.003`; Final `R2=0.8745`, MAE `1.047`
  - `Rhair`: Full `R2=0.8872`, MAE `3.865`; Final `R2=0.8385`, MAE `4.651`
  - `CO2air`: Full `R2=0.5086`, MAE `71.191`; Final `R2=0.4850`, MAE `72.943`
- 瀹稿弶鏌婃晶鐐叉彥闁喎顕В鏂挎禈閿?  - [hybrid_residual_vs_dlinear_joint_all_reference.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/comparisons/hybrid_residual_vs_dlinear_joint_all_reference.png)
  - 鐎规矮缍呴敍姘辨暏娴滃孩妲戞径鈺傜湽閹躲儲妞傝箛顐︹偓鐔风潔缁€琛♀偓婊冩躬閸氬奔绔?1-epoch quick benchmark 娑撳绱濆ǎ宄版値濞堝妯婂Ο鈥崇€烽惄绋款嚠 `DLinear` 閺勵垰鎯佸鑼病閸戣櫣骞囬弮鈺傛埂娴兼ê濞嶆穱鈥冲娇閳?- 瑜版挸澧犵拠缁樼《閿?  - 鏉╂瑧绮嶇紒鎾寸亯娴犲懏妲?1-epoch smoke test閿涘奔绗夐悽銊ょ艾濮濓絽绱＄紒鎾诡啈
  - 娴ｅ棗鐣犲鑼病鐠囦焦妲戦敍姝歨ybrid residual` 鏉╂瑦娼痪璺ㄦ畱娴狅絿鐖滈崗銉ュ經閵嗕浇顔勭紒鍐︹偓涔eckpoint閵嗕够ummary 閽€鐣屾磸闁棄鍑￠幍鎾烩偓姘剧礉閸欘垳娲块幒銉ф埛缂侇叀绐囧锝呯础妫板嫮鐣?  - 閸︺劌缍嬮崜?1-epoch quick benchmark 娑撳绱漙hybrid residual` 瀹歌尙绮￠崷?`Tair / CO2air` 娑撳﹥妲戦弰鍙ョ喘娴滃骸鎮撻崡蹇氼唴 `DLinear`閿涘矁鈧?`Rhair` 娑撳骸鍙鹃幒銉ㄧ箮娴ｅ棛鏆愰柅?  - 閺囨潙鎮庨悶鍡欐畱娑撳绔村銉︽Ц娑?`current_hybrid_transformer` 娴ｈ法鏁ら崥灞肩妫板嫮鐣婚敍鍫濐洤 `200 epoch, lr=1e-4, lambda_trend=0.3, patience=15`閿涘浠涘锝呯础鐎佃鐦敍灞藉晙閸愬啿鐣鹃弰顖氭儊閹恒儱鍙嗛幒褍鍩楁笟?benchmark
- 瀹歌尪藟閸嬫艾褰熸径鏍﹁⒈閺?residual 閸婃瑩鈧娈戦崥灞藉經瀵?1-epoch smoke test閿涘潉joint_all + Reference`閿涘argets = `Tair / Rhair / CO2air`閿涘绱?  - [itransformer_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/itransformer_residual_joint_all_reference_summary.json)
  - [patchtst_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/patchtst_residual_joint_all_reference_summary.json)
  - `iTransformer residual`
    - `Tair`: Full `R2=0.8447`, MAE `1.108`; Final `R2=0.8308`, MAE `1.141`
    - `Rhair`: Full `R2=0.8793`, MAE `4.249`; Final `R2=0.8359`, MAE `4.994`
    - `CO2air`: Full `R2=0.6084`, MAE `60.546`; Final `R2=0.5862`, MAE `61.666`
  - `PatchTST residual`
    - `Tair`: Full `R2=0.9244`, MAE `0.729`; Final `R2=0.9131`, MAE `0.783`
    - `Rhair`: Full `R2=0.8816`, MAE `4.004`; Final `R2=0.8619`, MAE `4.555`
    - `CO2air`: Full `R2=0.6422`, MAE `57.291`; Final `R2=0.6286`, MAE `58.864`
- 瑜版挸澧犳稉澶嬫蒋 residual 鐠侯垳鍤庨崷?1-epoch smoke test 娑撳娈戦弮鈺傛埂鐠囩粯纭堕敍?  - `Transformer-hybrid residual` 娴犲秵妲歌ぐ鎾冲閺堚偓瀵儤妫張鐔朵繆閸欏嚖绱濈亸銈呭従閸?`Tair / CO2air` 娑撳﹥娓堕弰搴㈡▔
  - `PatchTST residual` 閺勵垳娲伴崜宥嗘纯閸婄厧绶辩紒褏鐢婚惃鍕儑娴滃苯鈧瑩鈧绱濋弫缈犵秼瀹歌尙绮￠弰搴㈡▔娴兼ü绨崥灞藉礂鐠?`DLinear`閿涘奔绗栧В?`iTransformer residual` 閺囧菙
  - `iTransformer residual` 閸忋儱褰涘鍙夊ⅵ闁熬绱濇担鍡楃秼閸?1-epoch 娣団€冲娇閸嬪繐鎬ラ敍灞炬畯閺冩湹绗夋惔鏃€甯撻崷銊ュ娑撱倖娼稊瀣
  - 閸ョ姵顒濇潻娆忔噯閺囨潙鎮庨悶鍡欐畱濮濓絽绱℃０鍕暬閹恒劏绻樻い鍝勭碍鎼存柧璐熼敍?    - 閸忓牐绐?`Transformer-hybrid residual`
    - 閸愬秷绐?`PatchTST residual`
    - `iTransformer residual` 閺嗗倷绻氶悾娆庤礋缁楊兛绗侀崐娆撯偓?- `iTransformer residual` 瀹告彃鐣幋?`joint_all + Reference + 200 epoch fair-budget` 濮濓絽绱℃潻鎰攽閿?  - [itransformer_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/itransformer_residual_joint_all_reference_summary.json)
  - [itransformer_residual_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/itransformer_residual_joint_all_reference.pt)
  - `Tair`: Full `R2=0.9494`, MAE `0.618`; Final `R2=0.9362`, MAE `0.693`
  - `Rhair`: Full `R2=0.9030`, MAE `3.802`; Final `R2=0.8746`, MAE `4.412`
  - `CO2air`: Full `R2=0.7078`, MAE `51.161`; Final `R2=0.6947`, MAE `52.014`
- 瑜版挸澧犵€?`iTransformer residual` 濮濓絽绱＄紒鎾寸亯閻ㄥ嫯顕板▔鏇窗
  - 鐎瑰啫鎷?1-epoch 缂佹挻鐏夐惄鍛婄槷閹绘劕宕岄棃鐐茬埗閺勫孩妯夐敍宀冾嚛閺勫氦绻栭弶锛勫殠閺囩繝绶风挧鏍劀瀵繗顔勭紒鍐暕缁犳绱濇稉宥堝厴閻劍妫張?smoke test 鏉╁洦妫崥锕€鐣?  - 閸︺劌缍嬮崜宥嗩劀瀵繋绗侀弶?residual 娑擃叏绱漙iTransformer residual` 閺勵垱娓堕崸鍥€€閻ㄥ嫪绔撮弶鈽呯窗
    - `Rhair` 閺堚偓瀵?    - `CO2air` 娑旂喍绱禍搴″綗婢舵牔琚遍弶?    - `Tair` 閾忔垝绗夋俊?`Transformer-hybrid residual`閿涘奔绲炬禒宥勭箽閹镐浇绶濆?  - 閸ョ姵顒濊ぐ鎾冲閺囧菙閻ㄥ嫭顒滃蹇曠波鐠佸搫绨查弴瀛樻煀娑撶尨绱?    - `Transformer-hybrid residual` = 濞撯晛瀹抽張鈧?    - `PatchTST residual` = 濞嗏€茬喘娑撴柨顕?`CO2air` 閺堝鏁奸崰?    - `iTransformer residual` = 瑜版挸澧犳稉澶屾窗閺嶅洦鏆ｆ担鎾存付閸у洩銆€閵嗕焦娓堕崐鐓庣繁娴兼ê鍘涢幒銉ュ弳閹貉冨煑娓氀囩崣鐠囦胶娈?residual 閸婃瑩鈧?- `Transformer-hybrid residual` 瀹告彃鐣幋?`joint_all + Reference + 200 epoch fair-budget` 濮濓絽绱℃潻鎰攽閿?  - [transformer_hybrid_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/transformer_hybrid_residual_joint_all_reference_summary.json)
  - [transformer_hybrid_residual_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/transformer_hybrid_residual_joint_all_reference.pt)
  - `Tair`: Full `R2=0.9580`, MAE `0.526`; Final `R2=0.9502`, MAE `0.579`
  - `Rhair`: Full `R2=0.8210`, MAE `4.744`; Final `R2=0.7400`, MAE `5.945`
  - `CO2air`: Full `R2=0.6740`, MAE `55.315`; Final `R2=0.6310`, MAE `59.436`
- 瑜版挸澧犵€电绻栫紒鍕劀瀵繒绮ㄩ弸婊呮畱鐠囩粯纭堕敍?  - `Tair` 閺勫孩妯夐崣妯哄繁閿涘苯鍑＄紒蹇庣喘娴?1-epoch smoke test閿涘奔绡冩导妯圭艾閸氬瞼绮ㄩ弸鍕畱閺冣晜婀?quick benchmark
  - 娴?`Rhair / CO2air` 濞屸剝婀侀崥灞绢劄閸欐ɑ鍨氬娲€嶉敍宀冾嚛閺勫骸缍嬮崜?`Transformer-hybrid residual` 閻ㄥ嫭顒滃蹇氼唲缂佸啯鏁归惄濠佸瘜鐟曚線娉︽稉顓炴躬濞撯晛瀹虫稉鑽ゆ窗閺?  - 閸ョ姵顒濈€瑰啩绮涢悞鑸垫Ц瑜版挸澧?residual 娑撹崵鍤庨惃鍕繁閸婃瑩鈧绱濇担鍡氱箷娑撳秷鍏橀惄瀛樺复鐠併倕鐣炬稉琛♀偓婊€绗侀惄顔界垼閺佺繝缍嬮張鈧导妯封偓?  - 娑撳绔村銉ф埛缂侇厽顒滃蹇氱獓 `PatchTST residual` 娴犲秶鍔ч弰顖氱箑鐟曚胶娈戦敍灞芥礈娑撳搫鐣犻崷?1-epoch 娑撳娈?`Rhair` 娣団€冲娇閺囧瓨甯存潻鎴濆讲缁旂偘绨ら崠娲？
- `PatchTST residual` 瀹告彃鐣幋?`joint_all + Reference + 200 epoch fair-budget` 濮濓絽绱℃潻鎰攽閿?  - [patchtst_residual_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/patchtst_residual_joint_all_reference_summary.json)
  - [patchtst_residual_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/patchtst_residual_joint_all_reference.pt)
  - `Tair`: Full `R2=0.9440`, MAE `0.676`; Final `R2=0.9230`, MAE `0.829`
  - `Rhair`: Full `R2=0.8468`, MAE `4.991`; Final `R2=0.8121`, MAE `5.780`
  - `CO2air`: Full `R2=0.7311`, MAE `46.962`; Final `R2=0.6376`, MAE `55.862`
- 瑜版挸澧犵€?`PatchTST residual` 濮濓絽绱＄紒鎾寸亯閻ㄥ嫯顕板▔鏇窗
  - 閻╁憡鐦崗?1-epoch smoke test閿涘本顒滃蹇氼唲缂佸啫鎮?`CO2air` 閺勫孩妯夐崣妯哄繁閿涘本鍨氭稉鍝勭秼閸撳秷绻栭弶锛勫殠閺堚偓缁愪礁鍤惃鍕暪閻╁﹦鍋?  - `Rhair` 娑旂喍绱禍搴＄秼閸撳秵顒滃蹇曞 `Transformer-hybrid residual`
  - 娴?`Tair` 娴犲秵妲戦弰鎯ф€ユ禍搴＄秼閸撳秵顒滃蹇曞 `Transformer-hybrid residual`
  - 閸ョ姵顒濊ぐ鎾冲閺囧菙閻ㄥ嫮绮ㄧ拋杞扮瑝閺勵垪鈧粌鎽㈡稉鈧弶鈥冲弿鐠р懇鈧繐绱濋懓灞炬Ц閿?    - `Transformer-hybrid residual` 閺囧瓨鎼梹?`Tair`
    - `PatchTST residual` 閺囧瓨鎼梹?`Rhair / CO2air`
  - 閸?`iTransformer residual` 濮濓絽绱＄紒鎾寸亯閸戠儤娼甸崥搴礉`PatchTST residual` 閺囨潙鍣涵顔炬畱鐎规矮缍呮惔鏃囩殶閺佺繝璐熼敍?    - 鐎瑰啩绮涢悞鑸垫Ц閸氬牏鎮婇惃鍕儑娴滃本顫梼鐔封偓娆撯偓?    - 娴ｅ棗缍嬮崜宥嗘殻娴ｆ挸娼庣悰鈩冣偓褌绗夋俊?`iTransformer residual`
  - 鏉╂瑨绻樻稉鈧銉︽暜閹镐礁缍嬮崜宥勫瘜缁惧灝鍨介弬顓ㄧ窗鐎?AGC 鏉╂瑧琚径姘辨窗閺嶅洦甯堕崚璺侯嚤閸氭垳鎹㈤崝鈽呯礉娑撳秴鐡ㄩ崷銊ュ礋娑撯偓缂佹繂顕張鈧导妯肩波閺嬪嫸绱濆鐑樐侀崹瀣讲閼宠姤瀵滈惄顔界垼閸欐﹢鍣洪崚鍡楀
- 瀹稿弶鏌婃晶鐐电埠娑撯偓 residual 閸戝搫娴橀懘姘拱閿涙瓟plot_residual_forecaster_variants.py](c:/repositories/strawberry/agc_mpc/plot_residual_forecaster_variants.py)
  - 娴ｆ粎鏁ら敍姘槻閻劌甯慨?forecasting evaluator 閻ㄥ嫮鏁鹃崶楣冩懠鐠侯垽绱濇稉?residual 濮濓絽绱″Ο鈥崇€风悰銉╃秷娑?baseline 閻╃鎮撻弽鐓庣础閻ㄥ嫪绗佺猾璇叉禈閿?    - `forecast_examples`
    - `forecast_rollout`
    - `horizon_mae`
- 瑜版挸澧犳稉澶嬫蒋 residual 濮濓絽绱″Ο鈥崇€烽惃鍕禈閺傚洣娆㈠鎻掑弿闁劎鏁撻幋鎰剁窗
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
- 瀹稿弶鏌婃晶鐐寸垼閸戝棗鍤崶鎹愬壖閺堫剨绱癧plot_current_hybrid_transformer_standard.py](c:/repositories/strawberry/agc_mpc/plot_current_hybrid_transformer_standard.py)
  - 娴ｆ粎鏁ら敍姘礋 `current_hybrid_transformer` 婢跺秶鏁ょ紒鐔剁 evaluator閿涘矁藟姒绘劒绗?baseline / residual 閻╃鎮撻弽鐓庣础閻ㄥ嫪绗佺猾璇叉禈閿涘苯鑻熼幎?`figure_paths` 閸ョ偛鍟撻崚?summary
- `current_hybrid_transformer` 閺嶅洤鍣稉澶婃禈閻滄澘鍑＄悰銉╃秷閿?  - `horizon=24`
    - [current_hybrid_transformer_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_joint_all_reference_forecast_examples.png)
    - [current_hybrid_transformer_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_joint_all_reference_forecast_rollout.png)
    - [current_hybrid_transformer_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_joint_all_reference_horizon_mae.png)
  - `horizon=120`
    - [current_hybrid_transformer_h120_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h120_joint_all_reference_forecast_examples.png)
    - [current_hybrid_transformer_h120_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h120_joint_all_reference_forecast_rollout.png)
    - [current_hybrid_transformer_h120_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/current_hybrid_transformer/current_hybrid_transformer_h120_joint_all_reference_horizon_mae.png)



## 14. 最新控制侧 benchmark（2026-03-30，3-target latest predictor suite）

- 入口已扩展到 3-target 最新预测器：
  - `current_hybrid_transformer`
  - `transformer_hybrid_residual`
  - `itransformer_residual`
  - `patchtst_residual`
- 控制协议：`Reference + trajectory reference + surrogate rollout + 48 steps`
- 汇总 summary：
  - [latest_predictor_suite_reference_48steps.json](c:/repositories/strawberry/agc_mpc/results/control/summaries/latest_predictor_suite_reference_48steps.json)
- 汇总对比图：
  - [latest_predictor_suite_reference_48steps_metrics.png](c:/repositories/strawberry/agc_mpc/results/control/figures/latest_predictor_suite_reference_48steps_metrics.png)

### 14.1 current_hybrid_transformer
- `recorded`: `Tair=1.198`, `Rhair=3.409`, `CO2air=33.242`
- `GradientMPC`: `Tair=0.286`, `Rhair=0.921`, `CO2air=17.831`
- `CEMMPC`: `Tair=0.610`, `Rhair=1.662`, `CO2air=20.335`

### 14.2 transformer_hybrid_residual
- `recorded`: `Tair=0.462`, `Rhair=2.773`, `CO2air=26.300`
- `GradientMPC`: `Tair=0.472`, `Rhair=1.628`, `CO2air=16.128`
- `CEMMPC`: `Tair=1.013`, `Rhair=3.247`, `CO2air=20.591`

### 14.3 itransformer_residual
- `recorded`: `Tair=1.548`, `Rhair=4.661`, `CO2air=41.900`
- `GradientMPC`: `Tair=0.336`, `Rhair=2.587`, `CO2air=5.950`
- `CEMMPC`: `Tair=1.194`, `Rhair=4.906`, `CO2air=14.603`

### 14.4 patchtst_residual
- `recorded`: `Tair=2.403`, `Rhair=8.607`, `CO2air=40.093`
- `GradientMPC`: `Tair=1.047`, `Rhair=4.412`, `CO2air=17.127`
- `CEMMPC`: `Tair=1.986`, `Rhair=5.710`, `CO2air=23.904`

### 14.5 当前控制结论
- 在这组 3-target latest predictor suite 上，`GradientMPC` 仍然整体优于 `CEMMPC`。
- `current_hybrid_transformer + GradientMPC` 是当前最稳的整体闭环组合：`Tair / Rhair` 最好，且 objective mean 也最低梯队。
- `itransformer_residual + GradientMPC` 在 `CO2air` 上最强，`CO2air MAE=5.950`，明显优于另外三条预测器。
- `transformer_hybrid_residual` 的离线温度优势没有完整转化成闭环整体优势；控制侧表现更像温度和 CO2 之间的折中方案。
- `patchtst_residual` 在当前 surrogate 控制 benchmark 中整体最弱，说明它的离线 forecasting 改善没有顺利迁移到闭环控制收益。
- 当前最合理的控制主线排序：
  - `current_hybrid_transformer`
  - `itransformer_residual`
  - `transformer_hybrid_residual`
  - `patchtst_residual`
## 15. 正式同预算 DLinear 已补齐（2026-04-01）

- 之前用于和 residual 系列对比的 [dlinear_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/dlinear_forecaster_joint_all_reference_summary.json) 实际上是 `1 epoch` quick run，不能作为严格 fair-budget 结论。
- 现已按与 residual 系列一致的协议正式重跑 `DLinear`：
  - `joint_all + Reference`
  - targets = `Tair / Rhair / CO2air`
  - nominal budget = `200 epoch, lr=1e-4, lambda_trend=0.3, patience=15`
  - 实际 early stop 在最佳 `epoch 10`
- 正式同预算 DLinear 指标：
  - `Tair`: Full MAE `0.617`, Final MAE `0.682`
  - `Rhair`: Full MAE `3.944`, Final MAE `4.621`
  - `CO2air`: Full MAE `58.525`, Final MAE `61.130`
- 结果文件：
  - [dlinear_forecaster_joint_all_reference_summary.json](c:/repositories/strawberry/agc_mpc/results/forecasting/analysis/dlinear_forecaster_joint_all_reference_summary.json)
  - [dlinear_forecaster_joint_all_reference.pt](c:/repositories/strawberry/agc_mpc/results/forecasting/checkpoints/dlinear_forecaster_joint_all_reference.pt)
- 标准三图已补齐：
  - [dlinear_forecaster_joint_all_reference_forecast_examples.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_forecaster_joint_all_reference_forecast_examples.png)
  - [dlinear_forecaster_joint_all_reference_forecast_rollout.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_forecaster_joint_all_reference_forecast_rollout.png)
  - [dlinear_forecaster_joint_all_reference_horizon_mae.png](c:/repositories/strawberry/agc_mpc/results/forecasting/figures/baselines/dlinear_forecaster_joint_all_reference_horizon_mae.png)
- 基于这个正式 fair-budget 对照，当前结论更新为：
  - `transformer_hybrid_residual`：`Tair` 显著优于 DLinear，但 `Rhair / CO2air` 仍弱于正式 DLinear。
  - `itransformer_residual`：`Rhair / CO2air` 与正式 DLinear 接近或略优，但 `Tair` 略差于正式 DLinear。
  - `patchtst_residual`：整体尚未超过正式 DLinear。
  - 因此，`线性 + Transformer residual` 方向已经证明“有明显互补性”，但还不能说“现有混合模型已经全面超过正式同预算 DLinear”。

## 16. CO2 专项文献整理（2026-04-07）

- 已新增独立文档：
  - [CO2_PAPERS_AND_DIRECTION.md](c:/repositories/strawberry/agc_mpc/CO2_PAPERS_AND_DIRECTION.md)
- 这份文档主要回答两件事：
  - greenhouse / `CO2` 相关论文里的 `MAE` 是否归一化
  - 如果后续要专门改进 `CO2air`，哪些论文最值得读
- 当前从文献得到的稳定结论：
  - `CO2` 不是简单“再换一个 generic transformer backbone”就能稳住的问题。
  - 更常见、也更有效的方向是：
    - `CO2` 专项 `decomposition + recurrent/attention model + dynamic fusion`
    - `CO2 balance + photosynthesis + control` 的 gray-box 路线
- 对当前 `agc_mpc` 最现实的两条后续路线：
  - 路线 1：在现有 forecasting 架构内做 `CO2` 专项分支，重点尝试 decomposition 和 variable-weight fusion。
  - 路线 2：把 greenhouse 建成 `energy + water + carbon` 的 gray-box 系统，用 `CO2 dosing / ventilation exchange / canopy uptake` 等 latent flux 辅助建模。
