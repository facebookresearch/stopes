#! /bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# install npm / npx only for your user, not on the whole cluster
# as https://github.com/sindresorhus/guides/blob/main/npm-global-without-sudo.md

# Install nvm to update node
nvm install node


mkdir -p "${HOME}/.npm-packages"
npm config set prefix "${HOME}/.npm-packages"

echo '''NPM_PACKAGES="${HOME}/.npm-packages"

export PATH="$PATH:$NPM_PACKAGES/bin"

# Preserve MANPATH if you already defined it somewhere in your config.
# Otherwise, fall back to `manpath` so we can inherit from `/etc/manpath`.
export MANPATH="${MANPATH-$(manpath)}:$NPM_PACKAGES/share/man"
''' >> ${HOME}/.zshrc
