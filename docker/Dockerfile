FROM ubuntu:jammy
LABEL Author="whattheheckisthis" Description="A comprehensive docker image to run Apache-2.4 PHP-8.1 applications like Wordpress, Laravel, etc"


# Stop dpkg-reconfigure tzdata from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# Install apache and php7
RUN apt-get update && \
    apt-get -y install \
        apache2 \
        libapache2-mod-php \
        libapache2-mod-auth-openidc \
        php-bcmath \
        php-cli \
        php-curl \
        php-gd \
        php-intl \
        php-json \
        php-ldap \
        php-mbstring \
        php-memcached \
        php-mime-type \
        php-mysql \
        php-pgsql \
        php-soap \
        php-tidy \
        php-uploadprogress \
        php-xmlrpc \
        php-yaml \
        php-zip \
# Ensure apache can bind to 80 as non-root
        libcap2-bin && \
    setcap 'cap_net_bind_service=+ep' /usr/sbin/apache2 && \
    dpkg --purge libcap2-bin && \
    apt-get -y autoremove && \
# As apache is never run as root, change dir ownership
    a2disconf other-vhosts-access-log && \
    chown -Rh www-data. /var/run/apache2 && \
# Install ImageMagick CLI tools
    apt-get -y install --no-install-recommends imagemagick && \
# Clean up apt setup files
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
# Setup apache
    a2enmod rewrite headers expires ext_filter

# Override default apache and php config
COPY src/000-default.conf /etc/apache2/sites-available
COPY src/mpm_prefork.conf /etc/apache2/mods-available
COPY src/status.conf      /etc/apache2/mods-available
COPY src/99-local.ini     /etc/php/8.1/apache2/conf.d

# Expose details about this docker image
COPY src/index.php /var/www/html
RUN rm -f /var/www/html/index.html && \
    mkdir /var/www/html/.config && \
    tar cf /var/www/html/.config/etc-apache2.tar etc/apache2 && \
    tar cf /var/www/html/.config/etc-php.tar     etc/php && \
    dpkg -l > /var/www/html/.config/dpkg-l.txt


EXPOSE 80
USER www-data

ENTRYPOINT ["apache2ctl", "-D", "FOREGROUND"]
# Base image
FROM ubuntu:22.04

LABEL maintainer="Dhruv <itsdhruvsetty@gmail.com>"
LABEL description="Ops-Utilities Kernel + PHP/Apache Service (optional)"

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        apache2 libapache2-mod-php php-cli php-curl php-mbstring \
        python3.11 python3-pip python3-venv git bash zsh openssl \
        imagemagick && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Locale
RUN locale-gen en_US.UTF-8

# Python dependencies for ops-utilities
COPY config/requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip && pip install --no-cache-dir -r /tmp/requirements.txt

# Workdir
WORKDIR /workspace

# Copy root repo
COPY . .

# Make all scripts executable
RUN find scripts/ -type f -name "*.sh" -exec chmod +x {} \;

# Apache config (optional)
COPY src/000-default.conf /etc/apache2/sites-available/
COPY src/99-local.ini /etc/php/8.1/apache2/conf.d/
RUN a2enmod rewrite headers expires ext_filter

# User
USER www-data

EXPOSE 80
ENTRYPOINT ["apache2ctl", "-D", "FOREGROUND"]