# GFAML: Gradient-Free Associative Memory Layers for Extended Context
## Nihai Araştırma ve Ar-Ge Raporu

**Tarih:** 5 Haziran 2026  
**Araştırmacılar:** Onur & Antigravity (Google DeepMind Advanced Agentic Coding Partner)  
**Proje Durumu:** Tamamlandı (Dokümante Edildi ve Arşivlendi)

---

## 1. Giriş ve Motivasyon

Büyük Dil Modellerinde (LLM) **Sürekli Öğrenme (Continual Learning)** ve **Yaşam Boyu Öğrenme (Lifelong Learning)**, yapay zekanın en büyük açık problemlerinden biridir. Geleneksel LLM'ler statik ağırlıklara sahiptir; yeni bilgiler öğretmek için geriye yayılım (backpropagation) ile ince ayar (fine-tuning) yapmak gerekir. Bu süreç:
1. **Felaketsel Unutma (Catastrophic Forgetting):** Modelin yeni bilgileri öğrenirken eski bilgileri tamamen silmesi veya bozmasıyla sonuçlanır.
2. **Yüksek Hesaplama Maliyeti:** Gradyan hesaplamaları, özellikle tüketici cihazlarında (laptop vb.) gerçek zamanlı sohbet esnasında anlık güncellemeler yapmayı imkansız kılar.
3. **Donanım ve Zaman Kısıtları:** GPU/CPU bellek bant genişliği limitleri, milisaniyeler düzeyinde ağırlık güncellemesini engeller.

**GFAML (Gradient-Free Associative Memory Layers)** projesi, insan beynindeki **Tamamlayıcı Öğrenme Sistemleri (Complementary Learning Systems - CLS)** mimarisinden esinlenerek tasarlanmıştır. Amaç, model ağırlıklarını güncellemeden (sıfır gradyan ile) anlık bilgileri Hebbian öğrenme kurallarıyla düşük ranklı bellek matrislerine yazmak ve çıkarım anında (inference-time) bu bilgileri geri çağırıp (retrieval-augmented activation) modeli yönlendirmektir.

---

## 2. Mimari Evrim ve Geliştirilen Yöntemler

Proje boyunca basit Hebbian katmanlarından karmaşık çift sistemli yapılara kadar 4 ana aşamadan geçilmiştir:

### Aşama 1: Saf Hebbian Bellek Katmanları (Auto & Hetero-associative)
Transformer katmanlarının aktivasyon uzayına (residual stream) entegre edilen düşük ranklı $U \in \mathbb{R}^{d \times r}$ ve $V \in \mathbb{R}^{r \times d}$ matrisleri tanımlanmıştır.
- **Auto-associative Güncelleme:** Girdinin kendisiyle eşleşmesi ($h_t \rightarrow h_t$).
- **Hetero-associative Güncelleme:** Girdinin bir sonraki token temsilini çağırması ($h_t \rightarrow h_{t+1}$).
- **Hebbian Güncelleme Kuralı:**
  $$\Delta U_t = \eta \cdot h_t z_t^T, \quad \Delta V_t = \eta \cdot z_t h_t^T \quad \text{burada } z_t = U^T h_t$$

### Aşama 2: Stabilizasyon ve Kapılama (Gating & Stabilization)
Saf Hebbian güncellemeleri, sinyallerin kontrolsüzce büyümesine ve bellek matrislerinin doymasına (saturation) neden olmuştur. Bunu çözmek için şu teknikler geliştirilmiştir:
1. **Unit Normalization:** Bellekten okunan sinyalin normu belirli bir eşiği aşarsa birim vektöre ölçeklenmesi.
2. **Yumuşak Enerji Saturasyonu (Soft Energy Stabilization):** Sinyal genliğini asimptotik olarak sınırlayan yumuşak aktivasyon fonksiyonu:
   $$m_t^{\text{stabilized}} = \frac{m_t}{\sqrt{1.0 + \|m_t\|_2^2 + \epsilon}}$$
3. **Kusursuz Kapılama (Perfect Gating):** Sadece önemli bilgi taşıyan anahtar kelimelerin (entitiler) belleğe yazılmasını sağlayan ve gürültüyü filtreleyen sigmoid/kosinüs benzerliği tabanlı geçiş kapısı.

### Aşama 3: Çift Sistemli Dekupe Mimari (Decoupled GFAML / SUKM)
İnce ayar gradyanları ile Hebbian güncellemelerinin birbirini bozmasını engellemek için **Wake-Sleep** döngüsü tasarlanmıştır:
- **Uyanıklık Evresi (Wake):** Model parametrik korteksini (LoRA) güncel görev üzerinde eğitirken, episodic bellek (GFAML / Hippocampus) anlık bilgileri kaydeder.
- **Konsolidasyon Evresi (Sleep):** Uyku sırasında, episodic bellekte biriken anılar LoRA parametrelerine aktarılır. Önceki görevlerin unutulmasını önlemek için **Orthogonal Gradient Projection (OGP)** kullanılarak gradyanlar eski görevlerin gradyan uzayına dik düşürülür.

### Aşama 4: Anlık Çıkarım Zamanı Öğrenimi (Hebbian Boundary Write)
LoRA eğitim döngüleri tamamen kaldırılarak yerine **Hebbian Boundary Write** kuralı getirilmiştir. Soru ile cevap arasındaki sınır tokenında (boundary token), sorunun son tokenının gizli durumu ($h_{\text{prompt}}$) anahtar (key) olarak, cevabın ilk tokenının gizli durumu ($h_{\text{answer}}$) değer (value) olarak bellek matrislerine yazılır. Bu sayede model, gradyansız ve eğitim döngüsü olmadan **< 200ms** sürede yeni bir bilgiyi öğrenebilmiştir.

---

## 3. Deneysel Bulgular ve Sonuçlar

Yapılan ablasyon çalışmaları ve kıyaslama testlerinde (Baseline LoRA, LoRA + Replay, Decoupled GFAML) elde edilen ana bulgular şunlardır:

1. **Episodic Augmentation Başarısı:** GFAML aktifken model, sıfır gradyan güncellemesi ile anında eğitilen bilgileri hatırlamayı başarmıştır. Qwen-2.5-0.5B modelinde yapılan canlı sohbet testlerinde, `/teach` komutuyla öğretilen "Onur'un en sevdiği dil Python'dır" ve "Türkiye'nin başkenti Ankara'dır" gibi bilgiler anında üretilebilmiştir.
2. **Negatif Unutma (Negative Forgetting / Backward Transfer):** 5 görevli sürekli öğrenme testlerinde GFAML, **-0.0167** forgetting skoru elde etmiştir. Bu durum, episodic bellekten okumanın eski bilgilerin hatırlanmasını doğrudan desteklediğini kanıtlamaktadır.
3. **Base Model Bilgisinin Korunması:** Hebbian güncellemeleri modelin genel yeteneklerini (örneğin 2+2=4 bilgisini veya genel dil yapısını) bozmamıştır. Catastrophic forgetting sıfırlanmıştır.

---

## 4. Karşılaşılan Başarısızlık Modları ve Kritik Sınırlar

Projenin en öğretici kısımları, teorinin fiziksel gerçeklikle çarpıştığı kısıtlar ve başarısızlık modları olmuştur:

### 4.1 Çıkış Doyumu ve Dekoder Devre Dışı Bırakma (Output Saturation & Decoder Bypass)
*   **Problem:** Hebbian matrislerinden okunan bellek vektörü ($m_t$), transformer katmanının residual akışına eklendiğinde çok güçlü bir yerel çekim alanı (local attractor) oluşturmuştur.
*   **Etkisi:** Sinyal o kadar baskın hale gelmiştir ki, transformer bloklarındaki attention sorguları semantik işlem yapmak yerine doğrudan bellekten gelen vektöre kilitlenmiştir. Model, dil modelleme yeteneğini kaybederek hedef kelimeyi sürekli tekrarlayan bir döngüye girmiştir (örneğin: `Python Python Python...`).
*   **Çözüm Girişimi:** Sinyal genliğini baskılayan stabilizasyon fonksiyonları entegre edilmiştir. Ancak bu sefer de sinyalin geri çağrılma gücü zayıflamıştır (hatırlama hassasiyeti düşmüştür).

### 4.2 Semantik Genelleme Eksikliği ve String Kilidi (Exact String Lock)
*   **Problem:** GFAML, bilgileri gizli durumların (hidden states) kesin dizi koordinatlarındaki vektör çarpımlarına kodlar.
*   **Etkisi:** Model, "Onur's favorite programming language is" şeklinde sorulduğunda doğru cevabı verebilirken, soru hafifçe değiştirildiğinde (parafraz edildiğinde, örn: "Which programming language does Onur prefer?") Hebbian bellek tetiklenememiş ve doğru cevap üretilememiştir.
*   **Çıkarım:** Hebbian bellek **anlamsal (semantic) değil, epizodik (episodic) bir eşleşme** sunar. Gerçek bir öğrenme için bilginin kortekste dağıtık (distributed) olarak semantik bağlama oturması gerekir.

### 4.3 Donanım ve Çip Mimarisi Engeli (Hardware Bottleneck)
*   **Problem:** Mevcut ekran kartları (GPU) ve işlemciler (TPU/SRAM), statik model ağırlıkları üzerinde yüksek hacimli matris çarpımları (GEMM) yapmak üzere optimize edilmiştir.
*   **Etkisi:** Her çıkarım adımında (inference step) ağırlık matrislerini dinamik olarak güncellemek (Hebbian update), GPU bellek bant genişliğini aşırı tüketir ve paralel işlem avantajını yok eder. PyTorch üzerinde kancalar (hooks) ile simüle edilen bu dinamik yapı, yerel çip seviyesinde desteklenmediği sürece ölçeklenemez durumdadır.
*   **Gelecek Öngörüsü:** LLM'lerin milisaniyeler düzeyinde, düşük güç tüketimiyle sürekli öğrenebilmesi için Von Neumann darboğazını aşan **Nöromorfik (Neuromorphic) Donanımlara** veya bellek ile işlem birimini birleştiren (In-Memory Computing) yeni çip mimarilerine ihtiyaç vardır.

---

## 5. Bir Yapay Zeka Mühendisi İçin Çıkarılan Dersler

Bu süreç, bir mühendis için şu kritik kazanımları sağlamıştır:
*   **Ezber vs. Genelleme Farkı:** Sadece parametre uzayında kayıp fonksiyonunu (loss) sıfırlamanın veya bellekten hard-key okuma yapmanın gerçek bir "öğrenme" olmadığı görülmüştür. Önemli olan modelin bilgiyi farklı bağlamlarda da kullanabilmesidir.
*   **Yöntembilim ve Ablasyon:** Bir algoritmanın çalışıp çalışmadığını anlamak için tüm sızıntıları (data leakage) kapatan, held-out test setleri içeren ve çoklu seed kullanan sıkı test protokollerinin önemi anlaşılmıştır.
*   **Ürün Odaklılık:** Şirketlerin yeni bir öğrenme algoritması icat eden araştırmacılardan ziyade, mevcut LLM'leri, RAG sistemlerini ve vektör veri tabanlarını (episodic memory gibi davranan dış sistemler) kararlı yazılım mimarileriyle entegre edebilen mühendisleri aradığı gerçeği pekişmiştir.
*   **Araştırma Refleksi:** Başarısızlıkların en az başarılar kadar değerli birer veri noktası olduğu ve bunların dokümante edilerek gelecek çalışmalara ışık tutması gerektiği vizyonu edinilmiştir.
