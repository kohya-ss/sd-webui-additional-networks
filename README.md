## Additional Networks for generating images

日本語の文章は下のほうにあります。

## Updates / 更新情報

- Jan 8 2023, 2023/1/8: 
  - Supports medvram/lowvram in web UI. Thanks for ddvarpdd!
  - Web UIにmedvram/lowvramオプションを付けた場合でも動作するよう修正しました。ddvarpdd氏に感謝します。
- Jan 6 2023, 2023/1/6: 
  - Fixed a bug that broke the model were broken when switching enable->disable->enable...
  - SD 2.xのモデルで有効・無効を繰り返し切り替えるとモデルが壊れていく不具合を修正しました。
- Jan 5 2023, 2023/1/5: 
  - Added folder icon for easy LoRA selection. Fixed negative weights are not working.
  - ファイル選択ダイアログを開くアイコンを追加しました。負の重みを設定しても動かない不具合を修正しました。

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

Open __"Additional Networks"__ panel from the left bottom of Web UI.

Select __"LoRA"__ for __"Network module 1"__.

Enter __the full path of LoRA model file__ (*.ckpt or *.safetensors) in __"Model 1"__.

Set __the weight__ of the model (negative weight might be working but unexpected.)

Repeat them for the module/model/weight 2 to 5 if you have other models. Models are applied in the order of 1 to 5.

You can generate images with the model with these additional networks.

## このWeb UI拡張について

LoRAなどのネットワークを元のStable Diffusionに追加し、画像生成を行うための拡張です。現在はLoRAのみ対応しています。

この拡張で使えるのは[sd-scripts](https://github.com/kohya-ss/sd-scripts)リポジトリで学習したLoRAのモデル（\*.ckpt または \*.safetensors）です。他のLoRAリポジトリで学習したモデルは対応していません。

この拡張単体では学習はできません。

将来的にLoRA以外のネットワークについてもサポートするかもしれません。

## インストール

1. Web UIで "Extensions" タブを開きます。
1. さらに "Install from URL" タブを開きます。
1. "URL for extension's git repository" 欄にこのリポジトリのURLを入れます。
1. "Install"ボタンを押してインストールします。
1. Web UIを再起動してください。

## 使用法

Web UIの左下のほうの __"Additional Networks"__ のパネルを開きます。

__"Network module 1"__ で __"LoRA"__ を選択してください。

__"Model 1"__ に学習したLoRAのモデルファイルを __フルパス（ドライブ名やディレクトリ名付き）で__ 入力します。

__"Weight"__ にこのモデルの __重み__ を指定します（負の値も指定できますがどんな効果があるかは未知数です）。

追加のモデルがある場合は2～5に指定してください。モデルは1~5の順番で適用されます。

以上を指定すると、それぞれのモデルが適用された状態で画像生成されます。
