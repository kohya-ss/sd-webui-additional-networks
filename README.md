## Additional Networks for generating images

日本語の文章は下のほうにあります。

[__Change History__](#change-history) is moved to the bottom of the page.
更新履歴は[ページ末尾](#change-history)に移しました。

__Stable Diffusion web UI now seems to support LoRA trained by ``sd-scripts``__ (SD 1.x based only).  Thank you for great work!!!


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

## Change History

- 15 Feb. 2023, 2023/2/15
  - Fix a bug that additional networks are applied twice when restarting UI.
  - Now the cover image is not copied with ``Copy metadata`` in the metadata editor because it takes time.
  - Allow additional networks to be swapped out between batches for other extensions.
  - Thanks to space-nuko!
  - 各ネットワークがWeb UIの再起動後に二重に適用された状態になるバグを修正しました。
  - メタデータエディタで ``Copy metadata`` ボタンでカバー画像をコピーしなくなりました（処理に時間が掛かるため）。
  - バッチごとにネットワークを切り替えられるようになりました（他の拡張からLoRAを切り替えられます）。
  - space-nuko氏に感謝します。
- 12 Feb. 2023, 2023/2/12
  - __Dataset folder structure__ is shown in the metadata editor. Thanks to space-nuko!
  - メタデータエディタにデータセットのフォルダ構成が表示されるようになりました。space-nuko氏に感謝します。
- 10 Feb. 2023, 2023/2/10
  - Fixed a bug that crashes the metadata editor on some latest models.
  - 一部のモデルでメタデータエディタを開けない不具合を修正しました。
- 3 Feb. 2023, 2023/2/3
  - Tag frequency in training is shown in ``Training info`` in ``Additional Network`` tab. Thanks to space-nuko!
  - 学習時のタグ頻度情報が ``Additional Network`` タブの ``Training info`` に表示されるようになりました。space-nuko氏に感謝します。
- 2 Feb. 2023, 2023/2/2
  - This repo now has AGPL-3.0 license. Thanks to shirayu!
  - リポジトリにAGPL-3.0ライセンスが追加されました。取りまとめいただいたshirayu氏に感謝します。
- 1 Feb. 2023, 2023/2/1
  - Add ``send to metadata editor`` button in ``Additional Network`` in ``txt2img`` and other tabs. Thanks to space-nuko!
  - ``txt2img``タブ等にメタデータエディタに送るボタンが付きました。space-nuko氏に感謝します。
- 31 Jan. 2023, 2023/1/31
  - Metadata editor for LoRA models is now integrated in ``Additional Network`` tab. Documentation will be added later. Thanks to space-nuko!
  - LoRAモデル用のメタデータエディタ ``Additional Network`` タブに追加されました。ドキュメントはのちほど追加予定です。space-nuko氏に感謝します。

Please read [Releases](https://github.com/kohya-ss/sd-webui-additional-networks/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-webui-additional-networks/releases) をご覧ください。

