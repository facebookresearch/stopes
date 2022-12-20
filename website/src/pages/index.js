/**
 * Copyright (c) Meta Platforms, Inc. and affiliates
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import CodeBlock from '@theme/CodeBlock';
import Layout from '@theme/Layout';
import clsx from 'clsx';
import React from 'react';
import styles from './styles.module.css';

function Stopes() {
  return <span className={clsx('stopes')}>stopes</span>
}

const features = [
  {
    title: 'Easy to Use',
    description: (
      <>
        <Stopes /> was designed to provide a modular API to build and reproduce pipelines core to large translation work.
        In particular data mining and evaluation.
        Where you run your pipeline and how you scale it is independent of its core logic.
        Everything is config-driven so you can easily reproduce and track results.
      </>
    ),
    buttonTxt: "Quickstart",
    buttonUrl: 'docs/quickstart',
    imageUrl: 'img/shovel.svg',
  },
  {
    title: 'Batteries Included',
    description: (
      <>
        <Stopes /> lets you focus on your core data and evaluation needs by providing common modules
        used for this task and letting you write your pipelines with idiomatic python.
        Common optimizations have also been built-in to help you scale your work.
      </>
    ),
    buttonTxt: "Learn More",
    buttonUrl: 'docs/stopes',
    imageUrl: 'img/modules.svg',
  },
  {
    title: 'State-of-the-art Pipelines',
    description: (
      <>
        <Stopes /> was developed as part of the Meta AI No Language Left Behind research project.
        It comes with state-of-the-art pipelines out of the box. You can run our global mining and distillation
        pipelines and reproduce our research with just a few command lines.
      </>
    ),
    buttonTxt: "E.g. Start Data Mining",
    buttonUrl: 'docs/pipelines/global_mining',
    imageUrl: 'img/pipelines.svg',
  },
];


const sections = [
  {
    title: "No-coding Mining",
    language: 'bash',
    code: `python -m stopes.pipelines.bitext.global_mining_pipeline \\
   src_lang=fuv \\
   tgt_lang=zul \\
   demo_dir=./demo \\
   +preset=demo\\
   output_dir=. \\
   embed_text=laser3`,
    content: (
      <p><Stopes />  comes with the Global Mining Pipeline that was used by the NLLB team.
        You can use it out of the box without extra coding. You will need to setup an
        environment and create a config file to point to your data,
        but you can start mining (locally or on a slurm cluster) without any coding.
        Check out the <Link to="docs/quickstart">Quickstart guide</Link>.</p>
    )
  },
  {
    title: 'Reproducible research',
    language: 'yaml',
    code: `_target_: stopes.modules.preprocess.train_spm.TrainSpmModule
config:
  output_dir: ???
  vocab_size: 50_000
  input_sentence_size: 5_000_000
  character_coverage: 0.999995
  model_type: "unigram"
  shuffle_input_sentence: True
  num_threads : 4`,
    content: (
      <p>
        <Stopes /> is based on <Link to="http://hydra.cc/">Hydra</Link>,
        giving you full control over the behavior of your pipeline.
        Experiments are easily reproducible along with your results.</p>
    )
  },
  {
    title: 'Modular pipeline definition',
    language: 'python',
    code: `import asyncio

    import hydra
    from omegaconf import DictConfig
    from stopes.core.utils import clone_config
    from stopes.modules.bitext.indexing.populate_faiss_index import PopulateFAISSIndexModule
    from stopes.modules.bitext.indexing.train_faiss_index_module import TrainFAISSIndexModule

    # the pipeline
    async def pipeline(config):
        # setup a launcher to connect jobs together
        launcher = hydra.utils.instantiate(config.launcher)

        # train the faiss index
        trained_index = await launcher.schedule(TrainFAISSIndexModule(
            config=config.train_index
        ))

        # pass in the trained index to the next step through config
        with clone_config(config.populate_index) as config_with_index:
            config_with_index.index=trained_index

        # fill the index with content
        populated_index = await launcher.schedule(PopulateFAISSIndexModule(
            config=config_with_index
        ))
        print(f"Indexes are populated in: {populated_index}")

    # setup main with Hydra
    @hydra.main(config_path="conf", config_name="config")
    def main(config: DictConfig) -> None:
        asyncio.run(pipeline(config))
    `,
    content: (
      <>
        <p>
          <Stopes /> pipelines are composed of modules.
          No more duplicated, out-of sync code: your most common preprocessing steps can be shared
          among all your pipelines.
        </p>
        <p>
          You will find in this repository some implementations of a number of
          modules that are useful for translation data mining and evaluation, Neural Machine Translation data pre-processing
          and model training. For example, we provide modules to build <Link to="https://faiss.ai/">faiss</Link> indexes, encode
          text with <Link to="https://github.com/facebookresearch/LASER">LASER</Link> and <Link to="https://huggingface.co/sentence-transformers">HuggingFace Transformers</Link>,
          mine bitext or train and evaluate <Link to="https://github.com/facebookresearch/fairseq">FAIRSEQ</Link> models.
        </p></>
    )
  },
]

function Card({ title, description, buttonTxt, buttonUrl, imageUrl }) {
  const imgUrl = useBaseUrl(imageUrl);
  const burl = useBaseUrl(buttonUrl);

  return (
    <div className={clsx('col sfeatures')}>
      <div className={clsx("card card--full-height")}>
        {imgUrl && (
          <div className={clsx("card__image")}>
            <img
              src={imgUrl}
              alt={title}
              title={title} />
          </div>)}
        <div className={clsx("card__body")}>
          <h4>{title}</h4>
          <p>
            {description}
          </p>
        </div>
        {buttonTxt && buttonUrl && (
          <div className={clsx("card__footer")}>
            <Link
              className={clsx("button button--primary button--block")}
              to={burl}
            >
              {buttonTxt}
            </Link>
          </div>
        )}
      </div>
    </div>)
}

function ContentWithCode({ title, children, flip, language }) {

  const [content, code] = React.Children.toArray(children);

  const textBlock = (
    <div class="col col--4 scontent">
      {content}
    </div>
  )

  const codeBlock = (
    <div class="col">
      <CodeBlock language={language}>
        {code}
      </CodeBlock>
    </div>
  )

  let left = textBlock;
  let right = codeBlock;

  if (flip) {
    left = codeBlock;
    right = textBlock;
  }

  return (
    <div className="ssection">
      <div className="row">
        <h3>{title}</h3>
      </div>
      <div className="row">
        {left}
        {right}
      </div></div>)
}

function Banner() {


  const nllb = useBaseUrl('img/banner_bits/nllb.png');
  const driving = useBaseUrl('img/banner_bits/driving.png');
  const stopes = useBaseUrl('img/banner_bits/stopes.png');
  const meta = useBaseUrl('img/banner_bits/meta.png');
  const logo = useBaseUrl('img/logo.svg');

  return (
    <header className={clsx('sbanner shadow--md')}>
      <div className="gh-stars">
        <iframe
          src="https://ghbtns.com/github-btn.html?user=facebookresearch&amp;repo=stopes&amp;type=star&amp;count=true&amp;size=large"
          frameBorder={0}
          scrolling={0}
          width={160}
          height={30}
          title="GitHub Stars"
        />
      </div>
      <div className="container">
        <div className="sblue banner1">
          <img alt="NO LANGUAGES LEFT BEHIND" src={nllb} /></div>
        <div className="sblue banner2" >
          <img alt="Driving inclusion through machine translation" src={driving} /></div>
        <h1><img alt="logo" src={logo} className="logo" /><img alt="stopes" src={stopes} /></h1>
        <div className="banner3" >
          <h2>Large-Scale Translation Tooling</h2>
        </div>
      </div>
      <div className="bottom">
        <div className="button-container">
          <Link
            className={clsx(
              'button button--secondary button--lg',
            )}
            to={useBaseUrl('docs/quickstart')}>
            Mining Quickstart
          </Link>
        </div>
        <div className="banner-meta" >
          <img alt="meta" src={meta} />
        </div>
      </div>
    </header>
  )
}


export default function Home() {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Large-scale Translation Tooling">
      <Banner />
      <main className="container smain">
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div>
              <div className="row">
                {features.map(({ title, imageUrl, description, buttonTxt, buttonUrl }) => (
                  <Card
                    key={title}
                    title={title}
                    imageUrl={imageUrl}
                    description={description}
                    buttonTxt={buttonTxt}
                    buttonUrl={buttonUrl}
                  />
                ))}
              </div>
            </div>
          </section>
        )}
        <section>
          {sections.map(({ title, language, code, content }, index) => (
            <ContentWithCode
              key={title}
              flip={index % 2}
              language={language}
              title={title}>
              {content}
              {code}
            </ContentWithCode>
          ))}
        </section>
      </main>
    </Layout>
  );
}
