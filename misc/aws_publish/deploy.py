# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
This script was adapted from Dynabench deployer.py
All Dynabench dependency was stripped to make it standalone.

Also a reference implementation
https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-fastai-model-with-torchserve/blob/main/notebook/04_SageMaker.ipynb
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import logging
import os
import re
import shlex
import subprocess
import time
from functools import cached_property
from pathlib import Path
from typing import Any

import boto3
import botocore
import sagemaker

from deploy_config import config

logger = logging.getLogger("build")
logger.setLevel(logging.DEBUG)


class ModelDeployer:
    def __init__(
        self,
        src_dir: Path,
        name: str = "",
        use_gpu: bool = True,
        endpoint_name: str = "nllb200",
    ):
        self._src_dir = src_dir
        if not name:
            name = src_dir.name + datetime.datetime.now().strftime("-%y-%m-%d-%H-%M-%S")
            name = name.lower()
        assert re.match("[a-z0-9-]+", name), f"Name must be lowercase"
        self.name = name
        self.use_gpu = use_gpu

        self.instance_type = "ml.g4dn.4xlarge"
        self.instance_count = 2
        self.endpoint_name = endpoint_name
        self.endpoint_config_name = "-".join(
            (self.name, self.instance_type.split(".")[1])
        )

        self.archive_name = f"archive.{self.name}"
        self.full_archive_path = Path(
            f"/opt/ml/model/{src_dir.name}/{self.archive_name}.mar"
        )

        # self.rootp = tempfile.TemporaryDirectory(prefix=self.name)
        # self.root_dir = Path(self.rootp.name)
        # TODO: we should use a different dir to build the docker image from
        # deployer.py doesn't have src_dir
        # because the files are read from an archive from S3.
        self.root_dir = src_dir

    @cached_property
    def config(self) -> dict[str, Any]:
        """Returns the content of setup_config.json"""
        config_file = Path(__file__).parent / "dockerfiles" / "setup_config.json"
        return json.loads(config_file.read_text())

    def model_s3_path(self, split_bucket: bool = False) -> str:
        parts = (
            "s3:/",
            self.env["bucket_name"],
            "torchserve",
            f"{self.archive_name}.tar.gz",
        )
        if split_bucket:
            return "/".join(parts[2:])
        return "/".join(parts)

    @cached_property
    def image_ecr_path(self) -> str:
        return "/".join((self.env["ecr_registry"], self.name))

    def build_docker(self, lazy: bool = True) -> None:
        if lazy:
            image = subprocess.check_output(
                ["docker", "images", "-q", self.name], encoding="utf-8"
            )
            if image:
                logging.warning(
                    f"We already have a docker image for {self.name}, not rebuilding"
                )
                return
        # tarball current folder but exclude checkpoints
        tmp_dir = self.root_dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        config = self.config
        exclude_list = [
            "exclude.txt",
            "tmp",
            config["checkpoint"],
        ] + config.get("exclude", [])
        exclude_list_file = self.root_dir / "exclude.txt"
        exclude_list_file.write_text("\n".join(exclude_list))
        # Docker works with relative file paths
        tarball = f"tmp/{self.name}.tar.gz"
        subprocess.check_call(
            [
                "tar",
                "--exclude-from",
                exclude_list_file.name,
                "-czf",
                tarball,
                ".",
            ],
            cwd=self.root_dir,
        )

        docker_dir = Path(__file__).parent / "dockerfiles"
        for file in docker_dir.iterdir():
            cp(file, self.root_dir)

        setup_config = json.loads((docker_dir / "setup_config.json").read_text())
        # build docker
        build_args = {
            "tarball": tarball,
            "requirements": setup_config["requirements"],
            "task_code": setup_config["task"],
            "model_store": str(self.full_archive_path.parent),
            "model_name": self.full_archive_path.name,
        }
        docker_build_cmd = [
            "docker",
            "build",
            "--network",
            "host",
            "-t",
            self.name,
            "-f",
            str(docker_dir / "Dockerfile"),
            str(self.root_dir),
        ] + docker_build_args(**build_args)
        run_docker_cmd(docker_build_cmd)
        (self.root_dir / tarball).unlink()

    def push_docker(self) -> str:
        # TODO: detect if we need to push
        # docker login
        docker_credentials = self.env["ecr_client"].get_authorization_token()[
            "authorizationData"
        ][0]["authorizationToken"]
        user, p = base64.b64decode(docker_credentials).decode("ascii").split(":")
        subprocess.check_call(
            ["docker", "login", "-u", user, "-p", p, self.env["ecr_registry"]],
        )
        # tag docker
        image_ecr_path = self.image_ecr_path
        subprocess.check_call(["docker", "tag", self.name, image_ecr_path])

        # create ECR repo
        logger.info(f"Create ECR repository for model {self.name}")
        try:
            response = self.env["ecr_client"].create_repository(
                repositoryName=self.name,
                imageScanningConfiguration={"scanOnPush": True},
            )
        except self.env["ecr_client"].exceptions.RepositoryAlreadyExistsException as e:
            logger.debug(f"Reuse existing repository since {e}")
        else:
            if response:
                logger.debug(f"Response from creating ECR repository {response}")

        # push docker
        logger.info(f"Pushing docker instance {image_ecr_path} for model {self.name}")
        subprocess.check_call(["docker", "push", image_ecr_path])
        return image_ecr_path

    @cached_property
    def env(self) -> dict[str, Any]:
        region = config["aws_region"]
        account = config["account"]
        session = boto3.Session(
            aws_access_key_id=config["aws_access_key_id"],
            aws_secret_access_key=config["aws_secret_access_key"],
            aws_session_token=config["aws_session_token"],
            region_name=region,
        )

        e: dict[str, Any] = {}
        e["region"] = region
        e["sagemaker_client"] = session.client("sagemaker")
        e["s3_client"] = session.client("s3")
        e["ecr_client"] = session.client("ecr")

        e["bucket_name"] = sagemaker.Session(boto_session=session).default_bucket()
        e["ecr_registry"] = f"{account}.dkr.ecr.{region}.amazonaws.com"

        return e

    def archive_and_upload_model(self, lazy: bool = True) -> None:
        # torchserve archive model to .tar.gz (.mar)
        # TODO: allow proper model versioning together with docker tag
        if lazy:
            if path_available_on_s3(
                self.env["s3_client"],
                self.env["bucket_name"],
                self.model_s3_path(split_bucket=True),
            ):
                logger.warning(f"{self.model_s3_path()} exists, not re-uploading it")
                return

        mar = self.root_dir / f"{self.archive_name}.mar"
        archive_command = [
            "torch-model-archiver",
            "--model-name",
            self.archive_name,
            "--serialized-file",
            self.config["checkpoint"],
            "--handler",
            self.config["handler"],
            "--version",
            "1.0",  # TODO: proper versioning
            "--force",
        ]
        if self.config["model_files"]:
            extra_files = ",".join(
                shlex.quote(str(f)) for f in self.config["model_files"]
            )
            archive_command += ["--extra-files", extra_files]
        if lazy and mar.exists():
            logger.warning(f"{mar} exists, not re-archiving it")
        else:
            logger.info(f"Archiving the model {self.name} ...")
            logger.info(" ".join(shlex.quote(str(c)) for c in archive_command))
            subprocess.check_call(archive_command, cwd=self.root_dir)

        # tarball the .mar
        tarball = mar.with_suffix(".tar.gz")
        if lazy and tarball.exists():
            logger.warning(f"{tarball} exists, not re-archiving it")
        else:
            logger.info(f"Tarballing the archived model to {tarball}")
            subprocess.check_call(["tar", "cfz", tarball, mar])

        # upload model tarball to S3
        logger.info(
            f"Uploading the archived model to S3: {tarball} -> {self.model_s3_path()}"
        )

        response = self.env["s3_client"].upload_file(
            str(tarball),
            self.env["bucket_name"],
            self.model_s3_path(split_bucket=True),
        )
        if response:
            logger.info(f"Response from the mar file upload to s3 {response}")
        os.remove(tarball)
        os.remove(mar)

    def register_model(self):
        model_s3_path = "s3://" + self.model_s3_path()
        image_ecr_path = self.image_ecr_path
        logger.info(f"Deploying model {self.name} to Sagemaker")
        sm = self.env["sagemaker_client"]
        create_model_response = sm.create_model(
            ModelName=self.name,
            ExecutionRoleArn=config["arn_role"],
            EnableNetworkIsolation=True,
            PrimaryContainer={
                "Image": self.image_ecr_path,
                "ModelDataUrl": self.model_s3_path(),
            },
        )
        print(create_model_response)

    def create_endpoint_config(self):
        """
        Create a config corresponding to this model
        This make it easier to deploy the model through UI or script.
        It's safe to create the config, it's not deploying anything
        """
        self._create_endpoint_config(prod=False)
        self._create_endpoint_config(prod=True)

    def _create_endpoint_config(self, prod: bool):
        data_capture = {
            "DestinationS3Uri": f"s3://wiki-translate-queries/{self.name}/",
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        }
        extra_args = {"DataCaptureConfig": data_capture} if prod else {}
        sm = self.env["sagemaker_client"]

        create_endpoint_config_response = sm.create_endpoint_config(
            EndpointConfigName=prodname(self.endpoint_config_name, prod),
            ProductionVariants=[
                {
                    "ModelName": self.name,
                    "VariantName": self.name,
                    "InstanceType": self.instance_type,
                    "InitialInstanceCount": self.instance_count if prod else 1,
                    "InitialVariantWeight": 1.0,
                }
            ],
            Tags=[
                {"Key": "model", "Value": self.name},
                {"Key": "instance", "Value": self.instance_type},
                {"Key": "env", "Value": "prod"},
            ],
            **extra_args,
        )
        print(create_endpoint_config_response)

    def update_endpoint_config(self, prod: bool):
        sm = self.env["sagemaker_client"]
        update_endpoint_response = sm.update_endpoint(
            EndpointName=prodname(self.endpoint_name, prod),
            EndpointConfigName=prodname(self.endpoint_config_name, prod),
        )
        print(update_endpoint_response)

    def cleanup_post_deployment(self):
        try:
            # self.rootp.cleanup()
            # clean up local docker images
            subprocess.run(["docker", "rmi", self.name])
            image_tag = f"{self.env['ecr_registry']}/{self.name}"
            subprocess.run(["docker", "rmi", image_tag])
        except Exception as ex:
            logger.exception(f"Clean up post deployment for {self.name} failed: {ex}")


def prodname(name: str, prod: bool) -> str:
    if prod:
        return name
    return name + "-staging"


def docker_build_args(**build_args) -> list[str]:
    args = []
    for k, v in build_args.items():
        args.append("--build-arg")
        args.append(f"{k}={shlex.quote(str(v))}")
    return args


def run_docker_cmd(cmd: list[str], **kwargs) -> None:
    """Docker can be very verbous !"""
    logger = logging.getLogger("build.docker")
    logger.setLevel(logging.DEBUG)
    logger.info(" ".join(shlex.quote(str(c)) for c in cmd))
    with subprocess.Popen(
        cmd,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
        **kwargs,
    ) as process:
        while process.poll() is None:
            print(".", end="", flush=True)
            out = process.stdout.readline().rstrip("\n")
            while out:
                logger.debug(out)
                out = process.stdout.readline().rstrip("\n")
            time.sleep(10)
        stdout, stderr = process.communicate()
        logger.debug(stdout)
        print("!")

        if process.returncode != 0:
            logger.exception(f"Error in docker build: {cmd}")
            logger.info(stderr)
            raise RuntimeError("Error in docker build")


def cp(file: Path, target_dir: Path) -> None:
    (target_dir / file.name).write_bytes(file.read_bytes())


def path_available_on_s3(s3_client, s3_bucket, path, perturb_prefix=None):
    response = s3_client.list_objects_v2(Bucket=s3_bucket, Prefix=path)
    files = (obj["Key"] for obj in response.get("Contents", []))
    return path in files


def main(folder: Path, docker: str = "", prod: bool = False):
    """Make a docker image with the given folder and publish it to AWS as a model.
    A number of files are expected to be present in the folder,
    notably the setup_config.json, handler.py and checkpoint.pt

    You'll need to be allowed to talk to docker: `sudo usermod -aG docker $USER`
    Then logout and log back in.
    You'll also need AWS credentials.

    There are also a few Python dependencies you need to install with
    `pip install -r requirements.txt`
    """
    logging.basicConfig(level=logging.INFO)
    deployer = ModelDeployer(folder, name=docker)
    deployer.build_docker(lazy=True)

    try:
        deployer.env
    except Exception as e:
        logging.error(
            "Error while connecting to AWS, this can be due to expired credentials."
        )
        config["aws_access_key_id"] = input("AWS_ACCESS_KEY_ID:")
        config["aws_secret_access_key"] = input("AWS_SECRET_ACCESS_KEY:")
        config["aws_session_token"] = input("AWS_SESSION_TOKEN:")

    deployer.push_docker()
    deployer.archive_and_upload_model()
    deployer.register_model()
    deployer.create_endpoint_config()
    deployer.update_endpoint_config(prod=prod)
    deployer.cleanup_post_deployment()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        "-f", "--folder", type=Path, required=True, help="the folder with the model"
    )
    parser.add_argument(
        "-d",
        "--docker",
        default="",
        help="uses the given name for the docker image. Allows to skip docker build.",
    )
    parser.add_argument(
        "--prod",
        action="store_true",
        default=False,
        help="❗ deploys the model to prod (instead of staging env) ❗",
    )
    args = vars(parser.parse_args())
    main(**args)
