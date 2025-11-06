<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/selous123/al_sid">
    <img src="asset/Title.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">FORGE</h3>

  <p align="center">
     <b>FO</b>rming semantic identifie<b>R</b>s for <b>G</b>enerative retri<b>E</b>val in Industrial Datasets
    <br />
    <a href="https://huggingface.co/datasets/AL-GR"><strong>Explore full dataset in Huggingface »</strong></a>
    <br />
    <br />
    <a href="https://github.com/selous123/al_sid">View Demo</a>
    &middot;
    <a href="https://github.com/selous123/al_sid/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/selous123/al_sid/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details> -->



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/selous123/al_sid)

Semantic identifiers (SIDs) have gained increasing interest in generative retrieval (GR) due to their meaningful semantic discriminability. Existing studies typically rely on arbitrarily defined SIDs while neglecting the influence of SID configurations on GR. Besides, evaluations conducted on datasets with limited multimodal features and behaviors also hinder their reliability in industrial traffic. To address these limitations, we propose **FORGE**, a comprehensive benchmark for <u>**FO**</u>rming semantic identifie<u>**R**</u> in <u>**G**</u>enerative <u>**E**</u>trieval with industrial datasets. Specifically, FORGE is equipped with a dataset comprising **14 billion** user interactions and multi-modal features of **250 million** items collected from an e-commerce platform, enabling researchers to construct and evaluate their own SIDs.
Leveraging this dataset, we systematically explore various strategies for SID generation and validate their effectiveness across different settings and tasks. Extensive online experiments show **8.93\%** and **0.35\%** improvements in PVR and transaction count, highlighting the practical value of our approach. 
Notably, we propose two novel metrics of SID that correlate well with GR performance, providing insights into a convenient measurement of SID quality without training GR. Subsequent offline pretraining also offers support for online convergence in industrial applications. 
The code and data are available at [code repo](https://github.com/selous123/al_sid).

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- ### Built With

* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
1. Clone the repository:
   ```bash
   git clone repo_name
   cd repo
   ```
2. Install dependencies:
  * Python 3.8+
  * PyTorch 1.10+
  * requirements.txt
    ```sh
    pip install -r requirements.txt
    ```

<!-- ABOUT THE Dataset-->
### Dataset Decription(Demo)
1. For SID Generation Task:
```bash
wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/reconstruct_data_mask.npz
wget -P datas/ https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ICLR_2026_data/contrastive_data_mask.npz
```

Dataset Preview:
* 10m_80msideinfo_feat.npz for contrastive task in Eq.2

The file contains three components: 

(1) a deduplicated mapping table between item IDs and their corresponding indices ("itemEncId") with shape (6,844,930, 2), where each row is an [item_id, index] pair (e.g., [855036080309, 0]); 

(2) a list of item pairs ("pairs") with shape (9,509,084, 2), representing co-occurrence or association relationships between items (e.g., [855036080309, 545092516562]); and 

(3) a deduplicated embedding matrix with shape (6,844,930, 512), where each row is a 512-dimensional vector representation of an item (e.g., [xx, xx,..., xx]). 
```python
import numpy as np
import os

# 1. load .npz file
file_path = os.path.join(dirpath, filename2)  # replace your filename
data = np.load(file_path, allow_pickle=True)
itemEncID, pairs, embeds = data['itemEncID'].item(), data['pairs'], data['embeds'].astype(np.float32)
for key, item in itemEncID.items():
    print(key, item) #[855036080309, 0], [545092516562, 1]
    if item > 100:
        break
print("pairs:", pairs.shape, pairs[:1]) #(9509084, 2) [[855036080309 545092516562]]
print("embeds:", embeds.shape, embeds[:1]) #(6844930, 512) [[xx,xx,...,xx]]
```


* 5mold_80msideinfo_feat.npz for reconstruction task in Eq.3
``` python
import numpy as np
import os
# 1. load .npz file
dirpath = '~/git/al_sid/SID_generation/datas'
filename1 = '5mold_80msideinfo_feat.npz'
file_path = os.path.join(dirpath, filename1)  # replace your filename1
data = np.load(file_path)
print("Available arrays:", data.files) ##Available arrays: ['ids', 'embeds']
for key in data:
    print(f"{key}: {data[key].shape}")
    print(data[key][:1])
data.close()
#ids: (4148316,) [813799260043]
#embeds: (4148316, 512) [xx,xx,...,xx]
```


2. Seq Data for Generative Task:
```python
from datasets import load_dataset
# Login using e.g. `huggingface-cli login` to access this dataset
dataset = load_dataset("AL-GR/AL-GR-Tiny", data_files="train_data/s1_tiny.csv", split="train")
```

Data Preview: https://huggingface.co/datasets/AL-GR/AL-GR-Tiny

### SID Generation
1. Training the Model

To start distributed training, use the following command:
```bash
python -m torch.distributed.launch --nnodes=2 --nproc_per_node=1 --master_port=27646 train.py --output_dir=/path/to/output --save_prefix=MODEL_NAME --cfg=configs/rqvae_i2v.yml
```

2. Parameters
- `--cfg`: Path to the configuration file.
- `--output_dir`: Directory for model outputs.
- `--save_prefix`: Prefix for saving the model.

3. Testing the Model

Use the following command to start testing:

```bash
python infer_SID.py
```

### Generative Retrival

1. Clone the repo
   ```sh
   git clone repo_name
   cd repo_name/algr
   ```
2. training scripts:
    ```
    python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 runner.py --config=config/qwen2.5_05b_3layer_s1.json
    ```
3. predict scripts:
    ```
    python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 runner.py --config=config/generate_qwen2.5_05b_3layer_tiny.json
    ```
4. calculate Hitrate:
    ```
    # nebula test: python calc_hr.py --dataset_name=/home/admin/.cache/huggingface/modules/datasets_modules/datasets/AL-GR--AL-GR-Tiny/25dea07242891a2d --nebula
    python calc_hr.py --item_sid_file=item_info/tiny_item_sid_final.csv --generate_file=logs/generate_qwen2.5_05b_3layer_tiny/output.jsonl --decoder_only
    python calc_hr.py --item_sid_file=item_info/tiny_item_sid_final.csv --generate_file=logs/generate_t5base_3layer_tiny/output.jsonl
    ```
5. Running in the background
    ```
    nohup python -m torch.distributed.launch --nnodes=1 --nproc_per_node=16 runner.py --config=config/t5base_3layer_s1.json  >logs/t5base_3layer_s1/log.txt 2>&1 &
    nohup python -m torch.distributed.launch --nnodes=1 --nproc_per_node=16 runner.py --config=config/qwen2.5_05b_3layer_s1.json  >logs/qwen2.5_05b_3layer_s1/log.txt 2>&1 &
    ```


<!-- USAGE EXAMPLES -->
<!-- ## Usage
Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources. -->
<!-- _For more examples, please refer to the [Documentation]()_ -->
<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->


<!-- ROADMAP -->
## Roadmap

- [x] Generative Retrieval Training
- [x] SID Generation
- [x] Data Processing

See the [open issues](https://github.com/selous123/al_sid/issues) for a full list of proposed features (and known issues).

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->

### Top contributors:

<a href="https://github.com/selous123/al_sid/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=selous123/al_sid" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the project_license. See `LICENSE.txt` for more information.

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact
If you have any questions or encounter difficulties, we welcome you to contact ours via [GitHub Issues](https://github.com/selous123/al_sid/issues). We aim to respond promptly and support you in quickly getting up and running with generative recommendation.
<!-- Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com -->
<!-- Project Link: [https://github.com/selous123/al_sid](https://github.com/selous123/al_sid) -->
<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->


## Citing this work
Please cite the following paper if you find our code helpful.

```
@misc{fu2025forge,
      title={FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets}, 
      author={Kairui Fu and Tao Zhang and Shuwen Xiao and Ziyang Wang and Xinming Zhang and Chenchi Zhang and Yuliang Yan and Junjun Zheng and Yu Li and Zhihong Chen and Jian Wu and Xiangheng Kong and Shengyu Zhang and Kun Kuang and Yuning Jiang and Bo Zheng},
      year={2025},
      eprint={2509.20904},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2509.20904}, 
}
```


<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []() -->

<!-- <p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/selous123/al_sid.svg?style=for-the-badge
[contributors-url]: https://github.com/selous123/al_sid/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/selous123/al_sid.svg?style=for-the-badge
[forks-url]: https://github.com/selous123/al_sid/network/members
[stars-shield]: https://img.shields.io/github/stars/selous123/al_sid.svg?style=for-the-badge
[stars-url]: https://github.com/selous123/al_sid/stargazers
[issues-shield]: https://img.shields.io/github/issues/selous123/al_sid.svg?style=for-the-badge
[issues-url]: https://github.com/selous123/al_sid/issues
[license-shield]: https://img.shields.io/github/license/selous123/al_sid.svg?style=for-the-badge
[license-url]: https://github.com/selous123/al_sid/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: asset/FORGE.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
