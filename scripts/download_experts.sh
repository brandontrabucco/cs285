#!/bin/bash
curl https://codeload.github.com/berkeleydeeprlcourse/homework_fall2019/tar.gz/master | \
  tar -xz --strip=3 homework_fall2019-master/cs285/policies/experts -C $1/
