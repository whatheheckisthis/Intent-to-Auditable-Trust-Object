#!/bin/bash
# Tag the current commit with a version

VERSION=$1
MESSAGE=${2:-"Release $1"}
git tag -a "$VERSION" -m "$MESSAGE"
[ "$3" == "--push" ] && git push origin "$VERSION"
