"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[48],{3905:(e,t,n)=>{n.d(t,{Zo:()=>p,kt:()=>d});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var s=r.createContext({}),c=function(e){var t=r.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},p=function(e){var t=c(e.components);return r.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,s=e.parentName,p=l(e,["components","mdxType","originalType","parentName"]),m=c(n),d=a,h=m["".concat(s,".").concat(d)]||m[d]||u[d]||o;return n?r.createElement(h,i(i({ref:t},p),{},{components:n})):r.createElement(h,i({ref:t},p))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=m;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:a,i[1]=l;for(var c=2;c<o;c++)i[c]=n[c];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},8592:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>i,default:()=>u,frontMatter:()=>o,metadata:()=>l,toc:()=>c});var r=n(7462),a=(n(7294),n(3905));const o={},i="ALTI+",l={unversionedId:"eval/alti",id:"eval/alti",title:"ALTI+",description:"ALTI+ is a tool for inspecting token contributions in a transformer encoder-decoder model.",source:"@site/docs/eval/alti.md",sourceDirName:"eval",slug:"/eval/alti",permalink:"/stopes/docs/eval/alti",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/eval/alti.md",tags:[],version:"current",frontMatter:{},sidebar:"quickstartSidebar",previous:{title:"Evaluation Toolkit",permalink:"/stopes/docs/category/evaluation-toolkit"},next:{title:"BLASER: A Text-Free Speech-to-Speech Translation Evaluation Metric",permalink:"/stopes/docs/eval/blaser"}},s={},c=[],p={toc:c};function u(e){let{components:t,...n}=e;return(0,a.kt)("wrapper",(0,r.Z)({},p,n,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h1",{id:"alti"},"ALTI+"),(0,a.kt)("p",null,"ALTI+ is a tool for inspecting token contributions in a transformer encoder-decoder model.\nIt might be useful for detecting hallucinated translations or undertranslations."),(0,a.kt)("p",null,"This repository is based on the code from the paper ",(0,a.kt)("a",{parentName:"p",href:"https://arxiv.org/abs/2205.11631"},"Ferrando et al., 2022"),".\nThe original code is located at ",(0,a.kt)("a",{parentName:"p",href:"https://github.com/mt-upc/transformer-contributions-nmt"},"https://github.com/mt-upc/transformer-contributions-nmt"),".\nIt is licensed under the Apache 2.0 license included in the current directory."),(0,a.kt)("p",null,"We have made a few adaptation to the code so that it can run with the dense NLLB-200 models.\nThe code in this directory is licensed both under the Apache 2.0 license of the original code (in the current directory),\nand under the MIT license of the whole project (in the parent directory)."),(0,a.kt)("h1",{id:"usage"},"Usage"),(0,a.kt)("p",null,"An instruction for setting up the environment and computing ALTI+ token contributions from an NLLB model\nwith a command line interface is present in the folder ",(0,a.kt)("inlineCode",{parentName:"p"},"demo/alti"),"."),(0,a.kt)("p",null,"Below is another example, that uses a bilingual model and the Python interface.\nHere is how you can run it:"),(0,a.kt)("ol",null,(0,a.kt)("li",{parentName:"ol"},"Prepare the environment by installing Fairseq and Stopes:")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},"pip install fairseq==0.12.1\ngit clone https://github.com/facebookresearch/stopes.git\ncd stopes\npip install -e '.[alti]'\n")),(0,a.kt)("ol",{start:2},(0,a.kt)("li",{parentName:"ol"},"Download the model and dictionary from ",(0,a.kt)("a",{parentName:"li",href:"https://github.com/deep-spin/hallucinations-in-nmt"},"https://github.com/deep-spin/hallucinations-in-nmt"),":",(0,a.kt)("ul",{parentName:"li"},(0,a.kt)("li",{parentName:"ul"},"model: ",(0,a.kt)("a",{parentName:"li",href:"https://www.mediafire.com/file/mp5oim9hqgcy8fb/checkpoint_best.tar.xz/file"},"https://www.mediafire.com/file/mp5oim9hqgcy8fb/checkpoint_best.tar.xz/file")),(0,a.kt)("li",{parentName:"ul"},"data: ",(0,a.kt)("a",{parentName:"li",href:"https://www.mediafire.com/file/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz/file"},"https://www.mediafire.com/file/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz/file")))),(0,a.kt)("li",{parentName:"ol"},"Run the following commands to unpack the data:\n",(0,a.kt)("inlineCode",{parentName:"li"},"tar -xvf checkpoint_best.tar.xz && tar -xvf wmt18_de-en.tar.xz")),(0,a.kt)("li",{parentName:"ol"},"Run the following command to download the tokenizers:")),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre"},"wget https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.model\nwget https://github.com/deep-spin/hallucinations-in-nmt/raw/main/sentencepiece_models/sentencepiece.joint.bpe.vocab\n")),(0,a.kt)("p",null,"Now you can run the following Python code to look at the ALTI analysis:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-Python"},"from stopes.eval.alti.wrappers.transformer_wrapper import FairseqTransformerHub\nfrom stopes.eval.alti.alti_metrics.alti_metrics_utils import compute_alti_nllb, compute_alti_metrics\n\n# load the model, vocabulary and the sentencepiece tokenizer\nhub = FairseqTransformerHub.from_pretrained(\n    checkpoint_dir='.',\n    checkpoint_file='checkpoint_best.pt',\n    data_name_or_path='wmt18_de-en',\n    bpe='sentencepiece',\n    sentencepiece_model='sentencepiece.joint.bpe.model',\n)\n\n# translate an example of a German sentence to English.\n# the source sentence means \"The breakfast buffet is very good and varied.\", so the translation is wrong.\nsrc = 'Fr\xfchst\xfcckb\xfcffet ist sehr gut und vielseitig.'\ntgt = hub.translate(src)\nprint(tgt)  # The staff were very friendly and helpful.\n\n# compute the token contributions for this translation pair\nattributions, src_tok, tgt_tok, pred_tok = compute_alti_nllb(hub, src, tgt)\n# attributions is a 2d numpy array, and src/tgt/pred_tok are lists of subword strings\nprint(attributions.shape, len(src_tok), len(tgt_tok), len(pred_tok))  # (9, 21) 12 9 9\nprint(src_tok)  # ['\u2581Fr\xfchst\xfcck', 'b\xfc', 'ff', 'et', '\u2581ist', '\u2581sehr', '\u2581gut', '\u2581und', '\u2581vielseit', 'ig', '.', '</s>']\nprint(pred_tok)  # ['\u2581The', '\u2581staff', '\u2581were', '\u2581very', '\u2581friendly', '\u2581and', '\u2581helpful', '.', '</s>']\n\n# compute 18 different metrics based on the ALTI+ matrix.\n# 'avg_sc' is average source contribution, and the value of 0.4 is not very high (we expect about 0.5 or more).\nmetrics = compute_alti_metrics(attributions, src_tok, tgt_tok, pred_tok)\nprint(len(metrics))  # 18\nprint(metrics['avg_sc'])  # 0.40330514\n\n# for a correct translation, average source contribution is slightly higher\ntgt2 = \"The breakfast buffet is very good and diverse.\"\nprint(compute_alti_metrics(*compute_alti_nllb(hub, src, tgt2))['avg_sc'])  # 0.47343665\n")),(0,a.kt)("h1",{id:"citation"},"Citation"),(0,a.kt)("p",null,"If you use ALTI+ in your work, please consider citing:"),(0,a.kt)("pre",null,(0,a.kt)("code",{parentName:"pre",className:"language-bibtex"},"@inproceedings{alti_plus,\n    title = {Towards Opening the Black Box of Neural Machine Translation: Source and Target Interpretations of the Transformer},\n    author = {Ferrando, Javier and G\xe1llego, Gerard I. and Alastruey, Belen and Escolano, Carlos and Costa-juss\xe0, Marta R.},\n    booktitle = {Proc of the EMNLP},\n    url = {https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.599.pdf},\n    year = {2022}\n}\n")))}u.isMDXComponent=!0}}]);