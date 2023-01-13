## Additional Networks for generating images

日本語の文章は下のほうにあります。

## Updates / 更新情報

- Jan 13 2023, 2023/1/13:
  - The metadata for the model can be inspected from ``Additional Networks`` tab.
  - モデルのメタデータが  ``Additional Networks`` から確認できるようになりました。
- Jan 9 2023
  - The method of selecting a model has changed.
  - __Place models files in the folder  ``extensions\sd-webui-additional-networks\models\lora``__
    - All models, including in subfolders, are listed in the drop-down.
    - You can add an extra folder at ``Settings`` tab -> ``Additional Networks`` on left bottom -> ``Extra path to scan for LoRA models:``
  - Generated PNGs now have settings about LoRA in the infotext, which can be restored by sending it from ``PNG Info`` tab by ``txt2img`` button.
  - Thanks to space-nuko for this great contribution!

 - 2023/1/9:
    - モデル選択方法が変わりました。
    - __モデルファイルを Web UI の``extensions\sd-webui-additional-networks\models\lora`` フォルダに置いてください。__
      - サブフォルダにあるものも含めすべてのモデルがドロップダウンに表示されます。
      - スキャンするフォルダを追加できます。 ``Settings`` タブの ``Additional Networks`` （左下にあります）を選択し、 ``Extra path to scan for LoRA models:`` で設定してください。
    - 生成された PNG の infotext に設定が保存されるようになりました。  ``PNG Info`` タブから ``txt2img`` ボタンで復元されます。
    - 素晴らしいプルリクをいただいた space-nuko 氏に感謝します。

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
