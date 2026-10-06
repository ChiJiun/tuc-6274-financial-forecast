# raw/

此目錄保存**原始歷史資料**。

原則：

- raw 檔下載後不手動改值。
- 清洗、單位轉換、合併、特徵工程全部輸出到 `../processed/`。
- 每個 raw 檔應在 `manifest.csv` 留下來源 URL、下載時間、SHA-256 與檔案大小。
- MOPS 歷史月營收使用 **OTC** 路徑，因為台燿 6274 為上櫃公司。
- 預設模型只使用 2013-01 以後資料，但可保留 2003/12 起的歷史 raw 供研究 structural break。

主要目錄：

```text
raw/
├─ mops_monthly/
├─ investor_presentations/
├─ stockgo_6274_revenue.html
├─ moneydj_6274_revenue.html
├─ tuc_investor_relations.html
└─ manifest.csv
```

注意：原始文件的著作權與授權條款仍屬原始提供者，本 repo 的 MIT License 僅涵蓋程式碼。
