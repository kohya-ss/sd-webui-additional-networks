## Additional Networks for generating images

日本語の文章は下のほうにあります。

[__Change History__](#change-history) is moved to the bottom of the page.
更新履歴は[ページ末尾](#change-history)に移しました。

Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts`` Thank you for great work!!!


## About

This extension is for [AUTOMATIC1111's Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui), allows the Web UI to add some networks (e.g. LoRA) to the original Stable Diffusion model to generate images. Currently LoRA is supported. The addition is on-the-fly, the merging is not required.

This extension supports the LoRA models (*.ckpt or *.safetensors) trained by our scripts in [sd-scripts](https://github.com/kohya-ss/sd-scripts). The models from other LoRA implementations are not supported.

This extension does not support training.

Other networks other than LoRA may be supported in the future.

## Installation

1. Open "Extensions" tab.
1. Open "Install from URL" tab in the tab.
1. Enter URL of this repo to "URL for extension's git repository".
1. Press "Install" button.
1. Restart Web UI.

## How to use

Put the LoRA models (`*.pt`, `*.ckpt` or `*.safetensors`) inside the `sd-webui-additional-networks/models/LoRA` folder.

Open __"Additional Networks"__ panel from the left bottom of Web UI.

Press __"Refresh models"__ to update the models list.

Select __"LoRA"__ for __"Network module 1"__.

Choose __the name of the LoRA model file__ in __"Model 1"__.

Set __the weight__ of the model (negative weight might be working but unexpected.)

Repeat them for the module/model/weight 2 to 5 if you have other models. Models are applied in the order of 1 to 5.

You can generate images with the model with these additional networks.

## X/Y plot

If you use LoRA models to plot, put the comma separated list of the model names into ``AddNet Model X`` 

![image](https://user-images.githubusercontent.com/52813779/212444037-8ccd9157-c341-4eb4-82b4-64e3c8ee0237.png)

You can get the list of models with the button next to ``Values``. Please select any model in ``Model ?`` at ``Additional Networks`` in order to make the button work. Models in the same folder as the model will be listed.

![image](https://user-images.githubusercontent.com/52813779/212443639-97779d8d-0f7e-47f0-919c-41f053fe28a9.png)

The metadata of the model can be drawn as legends. Move to ``Settings`` tab, select ``Additional Networks`` at left bottom, and set ``Metadata to show``. Available values are in ``Network metadata`` textbox in ``Additional Networks`` tab.

![image](https://user-images.githubusercontent.com/52813779/212443781-1f4c442e-c2f3-47f8-b698-25fbe981f633.png)

## Specify target region of LoRA by mask (__experimental__)

Open `Extra args` and drop a mask image to `mask image`.

By specifying with the mask image, each LoRA model can be applied only to the specified region of the image. Currently, only three models (Models 1 to 3) can be masked.

The mask image is RGB image, with each channel (R, G and B) corresponding to LoRA models 1 to 3. Each channel can be overlapped. For example, yellow area (R and G) is applied to LoRA model 1 and 2. The range of values is 0 to 255, corresponding to a LoRA weight of 0 to 1.

It can be combined with ControlNet.

| |without ControlNet|with ControlNet|
|:----:|:----:|:----:|
|no LoRA|<img src="https://user-images.githubusercontent.com/52813779/223676928-362a68f0-b4c4-4905-9a5f-6646a39341f7.png" width="256">|<img src="https://user-images.githubusercontent.com/52813779/223677042-a7989dc8-741f-4d45-8328-be1f0bf08194.png" width="256">|
|with LoRA, no mask|<img src="https://user-images.githubusercontent.com/52813779/223677327-b4237ff9-1d36-4cd9-971b-a3434db6d0f9.png" width="256">|<img src="https://user-images.githubusercontent.com/52813779/223677380-ba74bca0-92c3-4c68-950f-0f96e439281e.png" width="256">|
|with Lora, with mask|<img src="https://user-images.githubusercontent.com/52813779/223677475-dff082c1-2a41-4d46-982d-db9655eb8bc2.png" width="256">|<img src="https://user-images.githubusercontent.com/52813779/223677518-0ae042ed-3baf-47f0-b8ca-3dd6805f7c2f.png" width="256">|
| |pose|mask|
| |<img src="https://user-images.githubusercontent.com/52813779/223677653-cfd7fb36-afc1-49e8-9253-4bc01c5dad99.png" width="256">|<img src="https://user-images.githubusercontent.com/52813779/223677672-5e2fc729-01ee-4c62-8457-2e125bb0e24f.png" width="256">

Sample images are generated with [wd-1-5-beta2-aesthetic-fp16.safetensors](https://huggingface.co/waifu-diffusion/wd-1-5-beta2) and three LoRAs: two character LoRAs (model 1 and 2, masked, weight=1.0) and one style LoRA (model 4, not masked, weight=0.8). Used ControlNet is [diff_control_wd15beta2_pose.safetensors](https://huggingface.co/furusu/ControlNet).

### Difference from 'Latent Couple extension' and 'Composable LoRA'

'Latent Couple extension' masks the output of U-Net for each sub-prompt (AND-separated prompts), while our implementation masks the output of LoRA at each layer of U-Net. The mask is resized according to the tensor shape of each layer, so the resolution is particularly coarse at the deeper layers.

'Composable LoRA' controls the area via 'Latent Couple extension' by switching LoRA on or off for each sub-prompt, but this implementation works alone.

This implementation does not work for all modules in LoRA (the modules associated with Text Encoder are not masked), and due to the coarse resolution, it is not possible to completely separate areas.

## この Web UI 拡張について

LoRA などのネットワークを元の Stable Diffusion に追加し、画像生成を行うための拡張です。現在は LoRA のみ対応しています。

この拡張で使えるのは[sd-scripts](https://github.com/kohya-ss/sd-scripts)リポジトリで学習した LoRA のモデル（\*.ckpt または \*.safetensors）です。他の LoRA リポジトリで学習したモデルは対応していません。

この拡張単体では学習はできません。

将来的に LoRA 以外のネットワークについてもサポートするかもしれません。

## インストール

1. Web UI で "Extensions" タブを開きます。
1. さらに "Install from URL" タブを開きます。
1. "URL for extension's git repository" 欄にこのリポジトリの URL を入れます。
1. "Install"ボタンを押してインストールします。
1. Web UI を再起動してください。

## 使用法

学習した LoRA のモデル(`*.pt`, `*.ckpt`, `*.safetensors`)を`sd-webui-additional-networks/models/LoRA`に置きます。

Web UI の左下のほうの __"Additional Networks"__ のパネルを開きます。

__"Network module 1"__ で __"LoRA"__ を選択してください。

__"Refresh models"__ で LoRA モデルのリストを更新します。

__"Model 1"__ に学習した LoRA のモデル名を選択します。

__"Weight"__ にこのモデルの __重み__ を指定します（負の値も指定できますがどんな効果があるかは未知数です）。

追加のモデルがある場合は 2～5 に指定してください。モデルは 1~5 の順番で適用されます。

以上を指定すると、それぞれのモデルが適用された状態で画像生成されます。

## X/Y plot

LoRAモデルをX/Y plotの値（選択対象）として使う場合は、カンマ区切りのモデルのリストを与える必要があります。

![image](https://user-images.githubusercontent.com/52813779/212444037-8ccd9157-c341-4eb4-82b4-64e3c8ee0237.png)

モデルのリストは選択肢の隣にあるボタンで取得できます。いずれかのモデルを ``Additional Networks`` の ``Model ?`` で選択しておいてください。そのモデルと同じフォルダにあるモデルの一覧が取得されます。

![image](https://user-images.githubusercontent.com/52813779/212443639-97779d8d-0f7e-47f0-919c-41f053fe28a9.png)

モデルのメタデータ（学習時のパラメータなど）をX/Y plotのラベルに使用できます。Web UI上部の ``Settings`` タブを開き、左下から ``Additional Networks`` を選び、 ``Metadata to show`` にカンマ区切りで項目名を指定してください（``ss_learning_rate, ss_num_epochs`` のような感じになります）。使える値は ``Additional Networks`` の ``Network metadata`` 欄にある値です。

![image](https://user-images.githubusercontent.com/52813779/212443781-1f4c442e-c2f3-47f8-b698-25fbe981f633.png)

## LoRA の領域別適用 __（実験的機能）__

適用する領域をマスク画像で指定することで、それぞれの LoRA モデルを画像の指定した部分にのみ適用することができます。現在はモデル1~3の3つのみ領域指定可能です。

マスク画像はカラーの画像で、RGBの各チャネルが LoRA モデル1から3に対応します。RGBの各チャネルは重ねることが可能です。たとえば黄色（RとGチャネル）の領域は、モデル1と2が有効になります。ピクセル値0から255がLoRAの適用率0から1に対応します（127なら重み0.5で適用するのと同じになります）。

マスク画像は生成画像サイズにリサイズされて適用されます。

ControlNetと組み合わせることも可能です（細かい位置指定にはControlNetとの組み合わせを推奨します）。

上のサンプルをご参照ください。

### Latent Couple extension、Composable LoRAとの違い

Latent Couple extension はサブプロンプト（ANDで区切られたプロンプト）ごとに、U-Net の出力をマスクしますが、当実装では U-Net の各層で LoRA の出力をマスクします。マスクは各層のテンソル形状に応じてリサイズされるため、深い層では特に解像度が粗くなります。

Composable LoRA はサブプロンプトごとに LoRA の適用有無を切り替えることで Latent Couple extension を経由して影響範囲を制御しますが、当実装では単独で動作します。

当実装はすべての LoRA モジュールに作用するわけではなく（Text Encoder に関連する LoRA モジュールはマスクされません）、また解像度が粗いため、完全に領域を分離することはできません。

## Change History

- 23 May 2023, 2023/5/23
  - Fix an issue where the value of the `Weight` slider is not applied correctly.
  - `Weight`のスライダーの値が正しく反映されない場合がある不具合への対応を行いました。
  
- 8 May 2023, 2023/5/8
  - Fix an issue where the models are not loaded correctly in the `Additional Networks` tab.
  - Fix an issue where `None` cannot be selected as a model in X/Y/Z plot.
  - `Additional Networks`タブでモデルが正しく読み込まれない不具合を修正しました。
  - X/Y/Z plotでモデルに `None` が選択できない不具合を修正しました。

- 3 May 2023, 2023/5/3
  - Fix an issue where an error occurs when selecting a model in X/Y/Z plot.
  - X/Y/Z plotでモデル選択時にエラーとなる不具合を修正しました。
- 6 Apr. 2023, 2023/4/6
  - Fix an issue where the `Hires. fix` does not work with mask.
  - 領域別LoRAでHires. fixが動作しない不具合を修正しました。
- 30 Mar. 2023, 2023/3/30
  - Fix an issue where the `Save Metadata` button in the metadata editor does not work even if `Editing Enabled` is checked.
  - メタデータエディタで `Save Metadata` ボタンが `Editing Enabled` をチェックしても有効にならない不具合を修正しました。
- 28 Mar. 2023, 2023/3/28
  - Fix style for Gradio 3.22. Thanks to space-nuko!
  - Please update Web UI to the latest version.
  - Gradio 3.22 のスタイルに対応しました。space-nuko氏に感謝します。
  - Web UIを最新版に更新願います。
- 11 Mar. 2023, 2023/3/11
  - Leading spaces in each path in `Extra paths to scan for LoRA models` settings are ignored. Thanks to tsukimiya!
  - 設定の `Extra paths to scan for LoRA models` の各ディレクトリ名の先頭スペースを無視するよう変更しました。tsukimiya氏に感謝します。
- 9 Mar. 2023, 2023/3/9: Release v0.5.1
  - Fix the model saved with `bf16` causes an error. https://github.com/kohya-ss/sd-webui-additional-networks/issues/127
  - Fix some Conv2d-3x3 LoRA modules are not effective. https://github.com/kohya-ss/sd-scripts/issues/275
  - Fix LoRA modules with higher dim (rank) > 320 causes an error.
  - `bf16` で学習されたモデルが読み込めない不具合を修正しました。 https://github.com/kohya-ss/sd-webui-additional-networks/issues/127
  - いくつかの Conv2d-3x3 LoRA モジュールが有効にならない不具合を修正しました。 https://github.com/kohya-ss/sd-scripts/issues/275
  - dim (rank) が 320 を超えるLoRAモデルが読み込めない不具合を修正しました。
- 8 Mar. 2023, 2023/3/8: Release v0.5.0
  - Support current version of [LoCon](https://github.com/KohakuBlueleaf/LoCon). __Thank you very much KohakuBlueleaf for your help!__
    - LoCon will be enhanced in the future. Compatibility for future versions is not guaranteed.
  - Support dynamic LoRA: different dimensions (ranks) and alpha for each module.
  - Support LoRA for Conv2d (extended to conv2d with a kernel size not 1x1).
  - Add masked LoRA feature (experimental.)
  - 現在のバージョンの [LoCon](https://github.com/KohakuBlueleaf/LoCon) をサポートしました。 KohakuBlueleaf 氏のご支援に深く感謝します。
    - LoCon が将来的に拡張された場合、それらのバージョンでの互換性は保証できません。
  - dynamic LoRA の機能を追加しました。各モジュールで異なる dimension (rank) や alpha を持つ LoRA が使えます。
  - Conv2d 拡張 LoRA をサポートしました。カーネルサイズが1x1でない Conv2d を対象とした LoRA が使えます。
  - LoRA の適用領域指定機能を追加しました（実験的機能）。


Please read [Releases](https://github.com/kohya-ss/sd-webui-additional-networks/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-webui-additional-networks/releases) をご覧ください。

