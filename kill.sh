kill -9 `ps ax |grep python3 | grep config | awk '{print $1}'`

