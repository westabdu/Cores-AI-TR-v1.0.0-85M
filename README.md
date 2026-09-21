# 🇹🇷 Cores-AI TR v1.0.0-85M

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Model Parameters](https://img.shields.io/badge/Parametre-81.17M-green.svg)](#model-mimarisi)
[![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-Model_Ağırlıkları-FFD21E.svg)](https://huggingface.co/aapo33/cores-ai-tr-v1.0.0-85m)
[![Vocab Size](https://img.shields.io/badge/Vocab-50%2C000-orange.svg)](#tokenizer)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> [!NOTE]
> **📢 GÜNCELLEME VE BİLGİLENDİRME (BASE MODEL & GELECEK PLANLARI):**
> - **Bu Model Base Model'dir:** Bu depo içerisindeki Cores-AI TR 85M modeli saf bir **Base Model** (ön eğitim / ham metin tamamlama modeli) olarak kalacaktır.
> - **SFT Durumu:** Bu model üzerinde şahsen SFT (Supervised Fine-Tuning) eğitimi gerçekleştirmeyi denedim; ancak 85M ölçeğindeki bu model beklediğim ve istediğim kıvama/performansa ulaşamadı. Bu nedenle şahsen **bu model için ek bir SFT (sohbet) eğitimi yapmayacağım.**
> - **Tamamen Açık Kaynak:** Kodlar ve ağırlıklar tamamen açık kaynaktır! İstediğiniz gibi mimariyi inceleyebilir, kodu geliştirebilir ve kendi SFT/fine-tuning denemelerinizi yapabilirsiniz.
> - **🚀 Yeni Modeller Geliyor:** Şu anda daha gelişmiş yeni modeller üzerinde aktif olarak çalışıyorum. Aklımda/geliştirme sürecinde **2 yeni model** var; bunlardan **1 tanesi hazırlandı, geriye yalnızca SFT eğitimi kaldı.** Yeni modelleri çok yakında ayrı projeler/repolar olarak duyuracağım, takipte kalın!

**Cores-AI TR**, tamamen Türkçe korpuslar üzerinde **sıfırdan (from scratch)** eğitilmiş, açık kaynaklı ve hafif siklet (81.17M parametre) bir Türkçe Üretken Dil Modelidir (GPT).

Bu depo; model mimarisini, özel Türkçe BPE tokenizer'ı, veri hazırlama ve ön eğitim (pre-training) betiklerini, ayrıca Google Colab T4 üzerinde yapılan eğitimin sonuçlarını ve metriklerini içerir.

---

## 🔗 Model Ağırlıklarını İndirme (Hugging Face)

Model ağırlıkları boyutu nedeniyle (978 MB) GitHub'da tutulmamaktadır. Ağırlık dosyasını (`ckpt.pt`) aşağıdaki Hugging Face bağlantısından indirebilirsiniz:

👉 **[Cores-AI-TR-85M Hugging Face Sayfası](https://huggingface.co/aapo33/cores-ai-tr-v1.0.0-85m)**

*İndirdiğiniz `ckpt.pt` dosyasını projeyi klonladıktan sonra `out-cores-85m-test/` klasörünün içine yerleştirmeniz gerekmektedir.*

---

## 🧠 Model Mimarisi

Cores-AI TR, modern Decoder-only Transformer mimarisine dayanmaktadır:

| Özellik | Değer |
|---|---|
| **Toplam Parametre Sayısı** | **81,165,440 (~81.17M)** |
| **Katman Sayısı (`n_layer`)** | 10 |
| **Dikkat Başlığı Sayısı (`n_head`)** | 10 |
| **Gömme Boyutu (`n_embd`)** | 640 |
| **Bağlam Uzunluğu (`block_size`)** | 512 token |
| **Kelime Dağarcığı (`vocab_size`)** | 50,000 (BPE Türkçe Tokenizer) |
| **Önyargı Terimi (`bias`)** | `False` (Daha verimli ve hızlı) |
| **Dropout** | 0.0 |
| **Aktivasyon Fonksiyonu** | GELU |

### 📦 Tokenizer
- **Dosya**: `tokenizer_tr_v2.json`
- Türkçe ek yapısına, ses uyumlarına ve özel karakterlere (ç, ğ, ı, ö, ş, ü vb.) göre optimize edilmiş **50.000 kelimelik Byte-Pair Encoding (BPE)** sözlüğü kullanılmıştır.
- `[BOS]` (Beginning of Sequence) ID: `2`

---

## 📊 Eğitim Detayları ve Metrikler

Model mimarisi, tokenizer ve veri akışı hattının doğrulanması amacıyla 5.000 adımlık eğitim gerçekleştirilmiştir.

### Eğitim Donanımı ve Parametreleri
- **Donanım**: Google Colab NVIDIA Tesla T4 (16 GB VRAM)
- **Hassasiyet**: FP16 (`float16`)
- **Optimizer**: AdamW (`lr=3e-4`, `min_lr=3e-5`, `weight_decay=0.1`, `beta1=0.9`, `beta2=0.95`)
- **Öğrenme Oranı Çizelgesi**: 500 adım Warmup + Cosine Decay
- **Toplu İş Boyutu (Batch Size)**: 16 (Adım başına gradyan biriktirme: 5 -> Efektif Batch = 80)
- **Adım Başına Token**: 40,960 token
- **Eğitim Verisi**: Türkçe Vikipedi + Matematik/Mantık Çözümleri + Türkçe Sohbet Veri Seti

### Kayıp (Loss) İlerlemesi
```text
Step 0    : Train Loss: 10.9320  |  Val Loss: 10.9370
Step 250  : Train Loss:  5.5882  |  Val Loss:  4.2955
Step 1000 : Train Loss:  4.3411  |  Val Loss:  3.7620
Step 2500 : Train Loss:  3.3105  |  Val Loss:  3.2104
Step 5000 : Train Loss:  2.7648  |  Val Loss:  2.8424  ✅ (En iyi val kaybı)
```

Loss değerinin 10.93 seviyesinden 2.84 seviyesine istikrarlı şekilde düşmesi, modelin Türkçe dil yapısını ve sözdizimini başarıyla kavramaya başladığını göstermektedir.

---

## 🧪 Örnek Test Çıktıları

Aşağıdaki örnekler, `sample_tr.py` betiği ile modelin gerçek ağırlıklarından (`ckpt.pt`) üretilmiştir:

### 1. Girdi: `"Türkiye"`
> **Model Çıktısı:**  
> *"Türkiye, Rusya, Özbekistan, Moldova, Tacikistan, İsrail, Belarus ve Nepal'de yaygın olarak kullanılan bir terimdir..."*  
> *(Diğer bir varyantta: Türkiye millî futbol takımları, lig şampiyonlukları ve şehir coğrafyası üzerine metinler üretmiştir.)*

### 2. Girdi: `"Yapay zeka"`
> **Model Çıktısı:**  
> *"Yapay zeka, her şeyin, insan ve bu bilim tarafından ölçülüğünü anlamak için gerekli olan bir sosyal bilimdir. Her şey, insan ve bu bilimde uygulanan sosyal bilimlerin araştırmalar için gerekli olan sonuçları, etik etkileri aracılığıyla bilimsel çalışmaları nasıl değerlendirdiğini belirlemeleri gerektiğini vurgular..."*

### 3. Girdi: `"İstanbul"` *(Matematik & Mantık Verisi Etkisi)*
> **Model Çıktısı:**  
> *"İstanbul'de 86 sandalye 3 kişiye eşit paylaştırılacak. Her kişi kaç sandalye alır?  
> Çözüm: 86 ÷ 3 = 28 sandalye (kalan: 2)  
> Örüntü: 25, 36, 49, 64, ? -> Sonraki: 81 (kareler: 9²)  
> Bir sayının 10 katının 3 fazlası 147 ise bu sayı kaçtır? Çözüm: Sayı = 14..."*

*(Not: Eğitim setindeki matematik soruları modelin bazı şehir promptlarında doğrudan problem kurgulama yeteneğini tetiklemiştir.)*

---

## 🚀 Kurulum ve Modeli Kullanma (Inference)

### 1. Depoyu İndirin ve Bağımlılıkları Kurun
```bash
git clone https://github.com/KULLANICI_ADINIZ/Cores-AI-TR-85M.git
cd Cores-AI-TR-85M
pip install -r requirements.txt
```

### 2. Ağırlıkları Yerleştirin
[Hugging Face deposundan](https://huggingface.co/aapo33/cores-ai-tr-v1.0.0-85m) `ckpt.pt` dosyasını indirin ve proje içerisindeki `out-cores-85m-test/` klasörüne (eğer klasör yoksa oluşturup içine) ekleyin.

### 3. Metin Üretimi (Inference)
Ağırlıklar yerleştirildikten sonra modeli aşağıdaki komutla test edebilirsiniz:

```bash
python sample_tr.py \
    --out_dir out-cores-85m-test \
    --tokenizer tokenizer_tr_v2.json \
    --start "Yapay zeka gelecekte" \
    --num_samples 3 \
    --max_new_tokens 150 \
    --temperature 0.8 \
    --top_k 200 \
    --device cuda
```

*GPU yoksa komutun sonundaki `--device cuda` kısmını `--device cpu` olarak değiştirebilirsiniz.*

---

## 🛠️ Kendi Modelini Eğitme

1. **Verileri Hazırlayın**:
   ```bash
   python prepare_tr.py
   ```
   *Bu işlem `data/turkce/` altına `train.bin`, `val.bin` ve `meta.pkl` dosyalarını üretir.*

2. **Eğitimi Başlatın**:
   ```bash
   python train.py config/train_cores_85m_test.py
   ```

---

## 📁 Depo Yapısı

```text
├── TESTLER/                  # Eğitim logları ve prompt test sonuçları
│   ├── ogrenme_verisi.txt    # 5.000 adımlık tam eğitim logu ve loss değerleri
│   └── prompt_testi.txt      # Farklı promptlarla üretilen ham metin çıktıları
├── config/                   # Model konfigürasyonları
│   └── train_cores_85m_test.py  # 85M model konfigürasyonu
├── configurator.py           # Komut satırından konfigürasyon override mekanizması
├── model.py                  # GPT / Decoder Transformer mimari tanımı
├── prepare_tr.py             # Hugging Face verilerini işleyip binary'e çeviren betik
├── sample_tr.py              # Metin üretim (inference) arayüzü
├── train.py                  # PyTorch DDP / AMP destekli ana eğitim motoru
├── tokenizer_tr_v2.json      # 50.000 kelimelik özel Türkçe BPE Tokenizer
├── requirements.txt          # Python kütüphane gereksinimleri
├── .gitignore                # Büyük binary / checkpoint dosyalarını filtreleme
└── LICENSE                   # MIT Açık Kaynak Lisansı
```

---

## 🗺️ Yol Haritası (Roadmap)

- [x] **Temel Model Eğitimi**
  - [x] Özel Türkçe BPE Tokenizer (50k) oluşturulması
  - [x] 85M parametreli modelin sıfırdan eğitimi
  - [x] Kaybın 10.93'ten 2.84'e düşürülmesi
- [ ] **Yeni Nesil Modeller & SFT**
  - [ ] Yeni yüksek parametreli modellerin eğitimi (1 model hazır, SFT aşamasında)
  - [ ] Yeni modeller için ayrı repo açılması ve duyurulması
- [ ] **Topluluk ve Açık Kaynak**
  - [ ] 85M Base model üzerinde topluluk fine-tuning çalışmalarına açık kaynak desteği

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) altında paylaşılmaktadır. Ticari ve kişisel kullanımda serbesttir.
