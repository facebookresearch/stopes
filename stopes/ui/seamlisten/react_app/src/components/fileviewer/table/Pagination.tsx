// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import { useRef } from "react";
import Button from "react-bootstrap/Button";
import { config } from "../../../common/constants/config";

type PaginProps = {
  page: number;
  setPage: Function;
};

function Pagination({ page, setPage }: PaginProps): JSX.Element {
  const refPage = useRef<HTMLInputElement>(null);

  function setComponentPage(x: number) {
    setPage(x);
    refPage.current.value = x.toLocaleString();
  }

  function changePage(i: number): void {
    let new_page = page + i;
    setComponentPage(new_page);
    refPage.current.value = new_page.toLocaleString();
  }
  function getPage(): number {
    return parseInt(refPage.current.value);
  }
  function submit(): void {
    setPage(getPage());
  }
  function handleKeyDown(event): void {
    if (event.key === "Enter") {
      submit();
    } else {
      if (!/[0-9]/.test(event.key)) {
        event.preventDefault();
      }
    }
  }

  return (
    <div style={{ margin: "auto", marginTop: "10px" }}>
      <Button
        style={{
          backgroundColor: config.default_color,
          borderColor: "white",
        }}
        onClick={() => {
          changePage(-1);
        }}
      >
        Previous page
      </Button>
      <input
        onKeyPress={handleKeyDown}
        defaultValue={page}
        ref={refPage}
        size={4}
      />
      <Button
        style={{
          backgroundColor: config.default_color,
          borderColor: "white",
        }}
        onClick={submit}
      >
        Go to!
      </Button>
      <Button
        style={{
          backgroundColor: config.default_color,
          borderColor: "white",
        }}
        onClick={() => {
          changePage(1);
        }}
      >
        Next page
      </Button>
    </div>
  );
}

export default Pagination;
