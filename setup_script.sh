# Setup /work as the working directory
sudo mkdir /work
sudo chmod -R 777 /work
cd /work

# Flask Setup
# Make sure the in the security group HTTP (port 80) is open in AWS
sudo yum -y install httpd24-devel
sudo pip install flask s3cmd findspark
mkdir /work/web_interface
sudo ln -sT /work/web_interface /var/www/html/web_interface
cat <<EOF > /work/web_interface/flaskapp.wsgi
import sys
sys.path.insert(0, '/var/www/html/web_interface')

from flaskapp import app as application
EOF
chmod a+x /work/web_interface/flaskapp.wsgi
# Put flaskapp.py in /work/web_interface
sudo -i
cat<<EOF >/etc/httpd/conf.d/vhost.conf
<VirtualHost *:80>
    WSGIDaemonProcess flaskapp threads=5
    WSGIScriptAlias / /var/www/html/web_interface/flaskapp.wsgi

    <Directory flaskapp>
        WSGIProcessGroup flaskapp
        WSGIApplicationGroup %{GLOBAL}
        Require all granted
    </Directory>
</VirtualHost>
EOF

cd ~
wget http://modwsgi.googlecode.com/files/mod_wsgi-3.4.tar.gz
tar xvf mod_wsgi-3.4.tar.gz
cd mod_wsgi-3.4
./configure --with-python=/usr/bin/python27
make
make install
LD_LIBRARY_PATH=/usr/lib64
ldconfig
# Add LoadModule wsgi_module modules/mod_wsgi.so to /etc/httpd/conf/httpd.conf
service httpd restart
exit;

# Install NLTK tools that are being used
sudo -u hdfs python2.7 -m nltk.downloader punkt
sudo -u hdfs python2.7 -m nltk.downloader stopwords
sudo -u apache mkdir /work/nltk_data
sudo -u apache python2.7 -m nltk.downloader punkt -d /work/nltk_data
sudo -u apache python2.7 -m nltk.downloader stopwords -d /work/nltk_data

# Setup for apache to run things on HDFS
sudo -u hdfs hdfs dfs -mkdir /user/apache
sudo -u hdfs hdfs dfs -chown apache:apache /user/apache

# Install and configure s3cmd
s3cmd --configure




# Submit paper_predictor.py setup in background
cd /work && nohup sudo -u hdfs spark-submit paper_predictor.py &