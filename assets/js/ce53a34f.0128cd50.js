"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[751],{3905:(e,t,n)=>{n.d(t,{Zo:()=>u,kt:()=>m});var a=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function r(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,a,i=function(e,t){if(null==e)return{};var n,a,i={},o=Object.keys(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(a=0;a<o.length;a++)n=o[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var s=a.createContext({}),p=function(e){var t=a.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):r(r({},t),e)),n},u=function(e){var t=p(e.components);return a.createElement(s.Provider,{value:t},e.children)},d={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},c=a.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,s=e.parentName,u=l(e,["components","mdxType","originalType","parentName"]),c=p(n),m=i,f=c["".concat(s,".").concat(m)]||c[m]||d[m]||o;return n?a.createElement(f,r(r({ref:t},u),{},{components:n})):a.createElement(f,r({ref:t},u))}));function m(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,r=new Array(o);r[0]=c;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:i,r[1]=l;for(var p=2;p<o;p++)r[p]=n[p];return a.createElement.apply(null,r)}return a.createElement.apply(null,n)}c.displayName="MDXCreateElement"},2266:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>s,contentTitle:()=>r,default:()=>d,frontMatter:()=>o,metadata:()=>l,toc:()=>p});var a=n(7462),i=(n(7294),n(3905));const o={sidebar_position:4},r="NLLB Distillation Pipeline",l={unversionedId:"pipelines/distillation",id:"pipelines/distillation",title:"NLLB Distillation Pipeline",description:"Welcome to stopes, and thanks for checking out our sequence-level knowledge distillation pipeline. This is a quick start guide which walks through how to run the pipeline yourself and what the expected outputs will be from each step. The logic of the pipeline is at a high level as follows:",source:"@site/docs/pipelines/distillation.md",sourceDirName:"pipelines",slug:"/pipelines/distillation",permalink:"/stopes/docs/pipelines/distillation",draft:!1,editUrl:"https://github.com/facebook/docusaurus/tree/main/packages/create-docusaurus/templates/shared/docs/pipelines/distillation.md",tags:[],version:"current",sidebarPosition:4,frontMatter:{sidebar_position:4},sidebar:"quickstartSidebar",previous:{title:"NLLB Monolingual Pipeline",permalink:"/stopes/docs/pipelines/monolingual"},next:{title:"Evaluation Toolkit",permalink:"/stopes/docs/category/evaluation-toolkit"}},s={},p=[{value:"To run:",id:"to-run",level:2},{value:"Useful overrides",id:"useful-overrides",level:2},{value:"Pipeline outputs",id:"pipeline-outputs",level:2},{value:"Raw input monolingual file:",id:"raw-input-monolingual-file",level:3},{value:"Example file output of monolingual_pipeline before dedup:",id:"example-file-output-of-monolingual_pipeline-before-dedup",level:3},{value:"Example file output of dedup",id:"example-file-output-of-dedup",level:3},{value:"Example file output of shard",id:"example-file-output-of-shard",level:3},{value:"Example file output of generate",id:"example-file-output-of-generate",level:3},{value:"Example file output of bitext clean",id:"example-file-output-of-bitext-clean",level:3},{value:"Example file output of binarizing and encoding",id:"example-file-output-of-binarizing-and-encoding",level:3},{value:"Example file output of train",id:"example-file-output-of-train",level:3}],u={toc:p};function d(e){let{components:t,...n}=e;return(0,i.kt)("wrapper",(0,a.Z)({},u,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"nllb-distillation-pipeline"},"NLLB Distillation Pipeline"),(0,i.kt)("p",null,"Welcome to ",(0,i.kt)("inlineCode",{parentName:"p"},"stopes"),", and thanks for checking out our sequence-level knowledge distillation pipeline. This is a quick start guide which walks through how to run the pipeline yourself and what the expected outputs will be from each step. The logic of the pipeline is at a high level as follows:"),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},"cleans pre-downloaded monolingual data (see ",(0,i.kt)("a",{parentName:"li",href:"https://github.com/fairinternal/nllb/blob/main/website/docs/pipelines/monolingual.md#nllb-monolingual-pipeline"},"STOPES monolingual pipeline"),") - results in one merged file of data for each source language"),(0,i.kt)("li",{parentName:"ol"},"shards each source language file from previous step into as many shards as number of specified target languages"),(0,i.kt)("li",{parentName:"ol"},"generates target language translations for each shard from previous step using Fairseq Generate"),(0,i.kt)("li",{parentName:"ol"},"cleans generated target language data and removes corresponding sentences from source language file used to generate translation"),(0,i.kt)("li",{parentName:"ol"},"binarizes and encodes the cleaned bitext data from previous step"),(0,i.kt)("li",{parentName:"ol"},"trains student model using the binarized distilled data from previous step")),(0,i.kt)("h2",{id:"to-run"},"To run:"),(0,i.kt)("p",null,"First, fill out any missing fields in distillation.yaml (labeled ???). Then,\n",(0,i.kt)("inlineCode",{parentName:"p"},"python stopes/pipelines/distillation/distillation_pipeline.py")," should be enough to get it running."),(0,i.kt)("p",null,"You can also override distillation.yaml fields manually through the CLI as such:\n",(0,i.kt)("inlineCode",{parentName:"p"},'python stopes/pipeliens/distillation/distillation_pipeline.py src_langs="[eng,mai]" tgt_langs="[fra,deu]" mono_data_dir=<path_to_predownloaded_mono_data> output_dir=<path_to_output_dir>'),"."),(0,i.kt)("p",null,"For internal FAIR users, feel free to add the ",(0,i.kt)("inlineCode",{parentName:"p"},"+fb_preset=nllb")," argument to the CLI command to use some preset config settings."),(0,i.kt)("p",null,"Note: Testing performance can be done with a separate STOPES module, ",(0,i.kt)("inlineCode",{parentName:"p"},"/stopes/modules/evaluation/generate_multi_bleu_detok_module.py"),"."),(0,i.kt)("h2",{id:"useful-overrides"},"Useful overrides"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},(0,i.kt)("inlineCode",{parentName:"p"},"src_langs")," is an array of source languages you have pre-downloaded monolingual data for")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},(0,i.kt)("inlineCode",{parentName:"p"},"tgt_langs")," is an array of target languages you want to train the student model to translate to")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},(0,i.kt)("inlineCode",{parentName:"p"},"mono_data_dir")," is the path to pre-downloaded monolingual data")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},(0,i.kt)("inlineCode",{parentName:"p"},"output_dir")," is the path to the desired output directory of this pipeline run")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},(0,i.kt)("inlineCode",{parentName:"p"},"skip_dedup=true launcher.cluster=local")," if you want to run this locally instead of on the slurm")),(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("p",{parentName:"li"},(0,i.kt)("inlineCode",{parentName:"p"},"launcher.cluster=slurm")," if you want to run this on the slurm instead of locally"))),(0,i.kt)("p",null,"See ",(0,i.kt)("inlineCode",{parentName:"p"},"distillation.yaml")," and it's associated config sub groups for more possible configuration options."),(0,i.kt)("h2",{id:"pipeline-outputs"},"Pipeline outputs"),(0,i.kt)("p",null,"Please be aware that at every intermediate step, the program will overwrite files with the same name (such as output from previous runs) so be sure to change the specified ",(0,i.kt)("inlineCode",{parentName:"p"},"output_dir")," or rename past outputs between runs."),(0,i.kt)("p",null,"The run will be started with a custom working directory that follows the pattern: ",(0,i.kt)("inlineCode",{parentName:"p"},"outputs/{date}/{start_time}"),", all the logs will go there (including executor_logs from slurm jobs). By default, the data output is set in ",(0,i.kt)("inlineCode",{parentName:"p"},"distillation.yaml")," to be ",(0,i.kt)("inlineCode",{parentName:"p"},"output_dir: .")," this means that the outputs will go to the working directory and will go to different places depending on the day/time you start the run. This is useful for testing, but if you want to output somewhere else (like a central clean monolingual repo), override the ",(0,i.kt)("inlineCode",{parentName:"p"},"output_dir=/somethingstable/")," when starting the run."),(0,i.kt)("h3",{id:"raw-input-monolingual-file"},"Raw input monolingual file:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"~/test_inputs/eng\n % cat test.eng\nAppealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture   http://www.sample_placeholder_website.url/  placeholder.gz  placeholder_sha 0\nno down payment auto insurance in Scottsdale AZ http://www.sample_placeholder_website.url/  placeholder.gz  placeholder_sha    847\nA Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac    50\n202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia  http://www.sample_placeholder_website.url/  placeholder.gz  placeholder_sha 8125\nBlackBerry Z10 To Launch In South Africa Tomorrow - Blackberry Empire   http://www.sample_placeholder_website.url/  placeholder.gz  placeholder_sha   0\n")),(0,i.kt)("h3",{id:"example-file-output-of-monolingual_pipeline-before-dedup"},"Example file output of monolingual_pipeline before dedup:"),(0,i.kt)("p",null,"Parsed in column format:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},'                1. self.corpus,  # the original corpus name\n                2. self.offset_start,  # skip that many bytes (use dd)\n                3. line_id,  # after skipping, go to line\n                4. line_hash,  # xxhash.xxh3_64 of the original line/paragrph\n                5. f"{prob_lang:.5f}",  # lid score\n                6. clean, # sentence\n                # config\n                sep="\\t"\n')),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"~/test_outputs/mono_data/eng\n % cat test.eng.000.sorted (processed and kept lines)\ntest    1056    0   4426603632439174366 0.71947 202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia\ntest    692 0   8327890826167111651 0.83095 A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac\ntest    0   0   12930410217004390762    0.90479 Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture\ntest    443 0   3451732902557484365 0.83896 no down payment auto insurance in Scottsdale AZ\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"~/test_outputs/mono_data/eng\n% cat _discarded.test.eng.000.sorted (discarded lines)\ntest    0   __label__eng    0.37420 BlackBerry Z10 To Launch In South Africa Tomorrow - Blackberry Empire\n")),(0,i.kt)("h3",{id:"example-file-output-of-dedup"},"Example file output of dedup"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"% cat eng_all_dedup\ntest    1056    0   4426603632439174366 0.71947 202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia\ntest    692 0   8327890826167111651 0.83095 A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac\ntest    0   0   12930410217004390762    0.90479 Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture\ntest2   0   0   5374428323341487497 1.00001 He has a cat.\ntest2   0   0   5374428323341487497 0.99987 Hello the president is here!\ntest    443 0   3451732902557484365 0.83896 no down payment auto insurance in Scottsdale AZ\n")),(0,i.kt)("h3",{id:"example-file-output-of-shard"},"Example file output of shard"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"% cat shard.000\ntest    1056    0   4426603632439174366 0.71947 202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia\ntest    692 0   8327890826167111651 0.83095 A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac\ntest    0   0   12930410217004390762    0.90479 Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture\ntest2   0   0   5374428323341487497 1.00001 He has a cat.\ntest2   0   0   5374428323341487497 0.99987 Hello the president is here!\ntest    443 0   3451732902557484365 0.83896 no down payment auto insurance in Scottsdale AZ\n")),(0,i.kt)("h3",{id:"example-file-output-of-generate"},"Example file output of generate"),(0,i.kt)("p",null,"Target generated data:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"test 1056 0 4426603632439174366 0.71947 202-458-1769 Joie Olverson - Spring House Ln, Washington, District de Columbia\ntest 692 0 8327890826167111651 0.83095 Une question de priorit\xe9s: r\xe9forme d\xe9mocratique et reprise \xe9conomique en Allemagne d'apr\xe8s-guerre Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac\ntest 0 0 12930410217004390762 0.90479 Chaise d'accent attrayante et chaise d'accent du tissu ottoman et du tissu de Petra avec meubles ottomans d\xe9coration de la maison - Lilangels meubles\ntest2 0 0 5374428323341487497 1.00001 Il a un chat.\ntest2 0 0 5374428323341487497 0.99987 Bonjour le pr\xe9sident est ici !\ntest 443 0 3451732902557484365 0,83896 aucun acompte d'assurance automobile \xe0 Scottsdale AZ\n")),(0,i.kt)("h3",{id:"example-file-output-of-bitext-clean"},"Example file output of bitext clean"),(0,i.kt)("p",null,"The contents of the filtered ",(0,i.kt)("inlineCode",{parentName:"p"},"clean.eng-fra.eng.000.xz")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"clean.eng-fra.fra.000.xz")," files are respectively:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"test    692 0   8327890826167111651 0.83095 A Question of Priorities: Democratic Reform and Economic Recovery in Postwar Germany Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac\ntest    0   0   12930410217004390762    0.90479 Appealing Accent Chair And Ottoman and Petra Fabric Accent Chair With Ottoman Furniture Home Decoration - Lilangels Furniture\ntest2   0   0   5374428323341487497 1.00001 He has a cat.\ntest2   0   0   5374428323341487497 0.99987 Hello the president is here!\ntest    443 0   3451732902557484365 0.83896 no down payment auto insurance in Scottsdale AZ\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"target_data 0   15  9889559120183218255 0.97691 Une question de priorit\xe9s: r\xe9forme d\xe9mocratique et reprise \xe9conomique en Allemagne d'apr\xe8s-guerre Auteur: Rebecca L. Boehling TiersD'occasion8,25\u20ac202,25\u20ac\ntarget_data 28  39  7358542291591603186 0.98684 Chaise d'accent attrayante et chaise d'accent du tissu ottoman et du tissu de Petra avec meubles ottomans d\xe9coration de la maison - Lilangels meubles\ntarget_data 56  61  4587081072824752671 0.81006 Il a un chat.\ntarget_data 56  68  13648239048374052831    0.99998 Bonjour le pr\xe9sident est ici !\ntarget_data 56  78  15942782228027469307    0.97898 aucun acompte d'assurance automobile \xe0 Scottsdale AZ\n")),(0,i.kt)("p",null,"Meanwhile, the contents of the two discarded output files ",(0,i.kt)("inlineCode",{parentName:"p"},"discarded.eng-fra.eng.000.xz")," and ",(0,i.kt)("inlineCode",{parentName:"p"},"discarded.eng-fra.fra.000.xz")," are respectively:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"test    1056    0   4426603632439174366 0.71947 202-458-1769 Joie Olverson - Spring House Ln, Washington, District of Columbia\n")),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"gen_shard   0   __label__eng    0.32102 202-458-1769 Joie Olverson - Spring House Ln, Washington, District de Columbia\n")),(0,i.kt)("h3",{id:"example-file-output-of-binarizing-and-encoding"},"Example file output of binarizing and encoding"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"train.eng-fra.eng.000.bin  train.eng-fra.eng.001.idx  train.eng-fra.eng.003.bin\ntrain.eng-fra.eng.000.idx  train.eng-fra.eng.002.bin  train.eng-fra.eng.003.idx\ntrain.eng-fra.eng.001.bin  train.eng-fra.eng.002.idx\n")),(0,i.kt)("h3",{id:"example-file-output-of-train"},"Example file output of train"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"-rw-rw-r-- 1 $USER $USER 4.2G Aug  3 12:05 checkpoint_best.pt\n-rw-rw-r-- 1 $USER $USER 4.2G Aug  3 12:05 checkpoint_last.pt\n")))}d.isMDXComponent=!0}}]);