# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import fire
import uvicorn


def main(setting="prod"):
    if setting == "prod":
        env_path = "prod.env"
        port = 8080
    else:
        env_path = "dev.env"
        port = 8000
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=port,
        env_file=env_path,
        reload=True,
        log_config="uvicorn_logging_config.yml",
    )


if __name__ == "__main__":
    fire.Fire(main)
