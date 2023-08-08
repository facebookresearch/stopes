// Copyright (c) Meta Platforms, Inc. and affiliates
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import "./App.css";
import Nav from "react-bootstrap/Nav";
import Navbar from "react-bootstrap/Navbar";
import { Outlet, NavLink } from "react-router-dom";

import "bootstrap/dist/css/bootstrap.min.css";
import Notebook, { Folder } from "./components/Notebook";

export function App(): JSX.Element {
  const folders : Folder[] = [
      {
        name: "folder1",
        type: "folder",
        content: [
          {
            name: "file1",
            type: "file",
            content: "file1 content"
          },
          {
            name: "file2",
            type: "file",
            content: "file2 content"
          },
          {
            name: "file3",
            type: "file",
            content: "file3 content"
          }
        ]
      },
      {
        name: "folder2",
        type: "folder",
        content: [
          {
            name: "file1",
            type: "file",
            content: "file1 content"
          },
          {
            name: "file2",
            type: "file",
            content: "file2 content"
          },
          {
            name: "file3",
            type: "file",
            content: "file3 content"
          }
        ]
      }
  ];
  return (
    <div className="App">
      <header>
        <Navbar bg="light" expand="lg">
          <Navbar.Brand href="/">
            <img src="/logo.png" height="60" alt="logo"></img>
          </Navbar.Brand>
          <Nav className="align-left mr-auto">
            <Nav.Item>
              <Nav.Link as={NavLink} to="/" className="align-left">
                File viewer
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link href="/docs" className="align-left">
                Interactive API
              </Nav.Link>
            </Nav.Item>
          </Nav>
        </Navbar>
      </header>
      <Outlet />
      <Notebook folders={folders} />
    </div>
  );
}

export default App;
