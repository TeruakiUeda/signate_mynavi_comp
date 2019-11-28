# signate_mynavi_comp

34位/302チーム中  
主な特徴量やモデル説明は下記記事参照  
solution https://tellmoogle.hatenablog.com/entry/signate_mynavicomp_losers_solution  

特徴量ごとにnotebookを分けて、作った特徴量はfeature_csv内にfeather形式で保存して、モジュール化     
validation setの作成やmoduleはcodeディレクトリ内の.pyで行う  
exp*.ipynbで特徴を選びながらLGBMを回す  
