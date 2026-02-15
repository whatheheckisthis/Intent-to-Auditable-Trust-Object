FROM php:8.3-fpm-alpine

WORKDIR /var/www/html

RUN apk add --no-cache bash

# Keep source ownership consistent for the www-data runtime user.
COPY --chown=www-data:www-data . /var/www/html

USER www-data

EXPOSE 9000
CMD ["php-fpm", "-F"]
