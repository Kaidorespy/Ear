# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ear_gui.pyw'],
    pathex=[],
    binaries=[],
    datas=[
        ('harsh_classifier/models/harsh_classifier.joblib', 'harsh_classifier/models'),
    ],
    hiddenimports=['sklearn.ensemble._forest', 'joblib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# Splash screen - shows while app loads
splash = Splash(
    'splash.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(188, 398),
    text_size=14,
    text_color='#f59e0b',
    text_default='Loading...',
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    a.binaries,
    a.datas,
    [],
    name='Ear',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['Ear.ico'],
)
