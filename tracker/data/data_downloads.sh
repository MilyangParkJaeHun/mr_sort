#!/bin/bash
# Downloads PxRx sequences data

OUT=train
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C7TBgL2aEhJ-iiesBLeoB4LGRM09kLCz' -O PmRm.tar.xz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1--0TgmvhNYm10-4VZGsAofrOjG6kvLW3' -O PmRm2.tar.xz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=16Ug0HZPmyxZ2XaAsO82MQMMDUBWBqDt2' -O PmRr.tar.xz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1n8JttvCr8R-kehT8uek3B9WQIecN-G5I' -O PoRt1.tar.xz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1GvzNIlrkxtDHxoBUM_YlFSctSAbgxmDF' -O PoRt2.tar.xz
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Pxjn7MrZkKhMojbgvorWo9Zk8icYJQPL' -O PsRm.tar.xz
tar -xvf PmRm.tar.xz
tar -xvf PmRm2.tar.xz
tar -xvf PmRr.tar.xz
tar -xvf PoRt1.tar.xz
tar -xvf PoRt2.tar.xz
tar -xvf PsRm.tar.xz
rm PmRm.tar.xz
rm PmRm2.tar.xz
rm PmRr.tar.xz
rm PoRt1.tar.xz
rm PoRt2.tar.xz
rm PsRm.tar.xz
mv P* $OUT
