# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/", tags=["root"])
async def read_root() -> FileResponse:
    return FileResponse("app/static/index.html")


@router.get("/logo.png", tags=["root"])
async def get_logo() -> FileResponse:
    return FileResponse("app/static/logo.png")
