#!/usr/bin/env bash
set -euo pipefail

FILES_DIR="/var/www/html/sites/default/files"
MOUNT_DIR="/mnt/s3-files"

mkdir -p "${FILES_DIR}" "${MOUNT_DIR}"
chown -R www-data:www-data "${FILES_DIR}"

if [[ -n "${S3_BUCKET:-}" ]]; then
  S3_URL="https://s3.${AWS_REGION:-us-east-1}.amazonaws.com"
  S3FS_OPTS=(
    -o use_path_request_style
    -o url="${S3_URL}"
    -o nonempty
    -o allow_other
    -o uid=33,gid=33,umask=0022
  )

  if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo "${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}" > /etc/passwd-s3fs
    chmod 600 /etc/passwd-s3fs
    S3FS_OPTS+=( -o passwd_file=/etc/passwd-s3fs )
  else
    S3FS_OPTS+=( -o iam_role=auto )
  fi

  if ! mountpoint -q "${MOUNT_DIR}"; then
    s3fs "${S3_BUCKET}:${S3_PREFIX:-sites/default/files}" "${MOUNT_DIR}" "${S3FS_OPTS[@]}"
  fi

  rm -rf "${FILES_DIR}"
  ln -sfn "${MOUNT_DIR}" "${FILES_DIR}"
fi

exec apache2-foreground
