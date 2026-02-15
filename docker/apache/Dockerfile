FROM httpd:2.4-alpine

RUN apk add --no-cache curl \
    && printf 'ok\n' > /usr/local/apache2/htdocs/healthz

COPY docker/apache/httpd.conf /usr/local/apache2/conf/httpd.conf
