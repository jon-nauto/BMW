# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)
block_cipher = None

datas = [('models\encrypted\mobilenet_v1-channels_first.json.encrypted','.'),('models\encrypted\mobilenet_v1-channels_first.h5.encrypted','.'),('visualization.py','.'),('utils.py','.')]
a = Analysis(['BMWDemo.py'],
             pathex=[],
             binaries=[],
             datas=datas,
             hiddenimports=['keras'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='please',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
