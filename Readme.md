# Single Image Super Resolution

This repo contains the implementations of some common models used for SISR. 

For simple inference tasks, download the pretrained models from [here][pretrained]. Place them in the ```pretrained/``` directory.

Next, configure the yaml config file in ```configs/``` directory for the respective model you wish to run and then run inference as:

```
python main.py --config-file <config file>
```
For example:
```
python main.py --config-file RCAN.yaml
```

## Report
A complete summary report of these models can be found here: [report_link][report_link]

## References:

* Enhanced Deep Super Resolution Network (EDSR) : [link][edsr_link]
* Residual Channel Attention Network (RCAN) : [link][rcan_link]


[pretrained]: https://drive.google.com/drive/folders/1vSvHM_Bj0ZwFTU6MJSkYhDn64VA-S8-U?usp=sharing
[report_link]: https://drive.google.com/file/d/1qCeQg06F6w7UNK-cVGsk7h282v19dybr/view?usp=sharing
[edsr_link]: https://arxiv.org/abs/1707.02921
[rcan_link]: https://arxiv.org/abs/1807.02758

Note: The code for GAN based models is not currently released. It will be released later.