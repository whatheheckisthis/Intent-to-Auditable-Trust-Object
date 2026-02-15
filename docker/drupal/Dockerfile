FROM drupal:10-apache

RUN apt-get update \
    && apt-get install -y --no-install-recommends s3fs curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY docker/drupal/drupal-s3-entrypoint.sh /usr/local/bin/drupal-s3-entrypoint.sh
RUN chmod +x /usr/local/bin/drupal-s3-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/drupal-s3-entrypoint.sh"]
