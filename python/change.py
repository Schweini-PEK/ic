
import io,sys
p='F:/new/ic/src/factor/symbolic/pattern.cpp'  # 改成你的路径
s=open(p,'rb').read()
# try remove BOM
if s.startswith(b'\xef\xbb\xbf'):
    s=s[3:]
open(p,'wb').write(s)
print('saved no-bom')
