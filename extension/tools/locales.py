"""
Source of the UI strings; writes _locales/<lang>/messages.json (chrome.i18n format).
Edit the table below, then run `python3 extension/tools/locales.py`.
"""

import json
from pathlib import Path

EXT = Path(__file__).resolve().parent.parent
LANGS = ("en", "tr", "az", "ru")

# key: (en, tr, az, ru)
STRINGS = {
    "extName": ("Captchosa", "Captchosa", "Captchosa", "Captchosa"),
    "extDescription": (
        "Solves AWS WAF, GeeTest, reCAPTCHA v2 image, BLS and text captchas with your own self-hosted Captchosa service.",
        "AWS WAF, GeeTest, reCAPTCHA v2 resim, BLS ve metin captcha'larını kendi Captchosa sunucunuzla çözer.",
        "AWS WAF, GeeTest, reCAPTCHA v2 şəkil, BLS və mətn captcha-larını öz Captchosa serverinizlə həll edir.",
        "Решает капчи AWS WAF, GeeTest, reCAPTCHA v2 (картинки), BLS и текстовые через ваш собственный сервис Captchosa."),
    "on": ("On", "Açık", "Aktiv", "Вкл."),
    "off": ("Off", "Kapalı", "Deaktiv", "Выкл."),
    "masterSwitch": ("Turn Captchosa on or off", "Captchosa'yı aç/kapat", "Captchosa-nı aç/bağla", "Включить или выключить Captchosa"),
    "solvers": ("Solvers", "Çözücüler", "Həlledicilər", "Решатели"),
    "kind_awsGrid": ("AWS WAF grid", "AWS WAF ızgara", "AWS WAF şəbəkə", "AWS WAF сетка"),
    "kind_awsToken": ("AWS WAF token", "AWS WAF token", "AWS WAF token", "AWS WAF токен"),
    "kind_blsGrid": ("BLS grid", "BLS ızgara", "BLS şəbəkə", "BLS сетка"),
    "kind_blsText": ("BLS Poland text", "BLS Polonya metin", "BLS Polşa mətn", "BLS Польша текст"),
    "kind_text": ("Text captcha", "Metin captcha", "Mətn captcha", "Текстовая капча"),
    "kind_geetest": ("GeeTest", "GeeTest", "GeeTest", "GeeTest"),
    "kind_recaptcha": ("reCAPTCHA v2", "reCAPTCHA v2", "reCAPTCHA v2", "reCAPTCHA v2"),
    "desc_awsGrid": ("Picture grid in the AWS WAF widget", "AWS WAF penceresindeki resim ızgarası",
                     "AWS WAF pəncərəsindəki şəkil şəbəkəsi", "Сетка картинок в виджете AWS WAF"),
    "desc_awsToken": ("Token for the AWS challenge page", "AWS doğrulama sayfası için token",
                      "AWS yoxlama səhifəsi üçün token", "Токен для страницы проверки AWS"),
    "desc_blsGrid": ("“Select all boxes with number …”", "“… numaralı tüm kutuları seçin”",
                     "“… nömrəli bütün qutuları seçin”", "«Выберите все ячейки с числом …»"),
    "desc_blsText": ("5 digits on the BLS Poland login", "BLS Polonya girişindeki 5 rakam",
                     "BLS Polşa girişindəki 5 rəqəm", "5 цифр при входе BLS Польша"),
    "desc_text": ("Any image text captcha, BLS Poland included (detected automatically)",
                  "BLS Polonya dahil her resimli metin captcha (otomatik algılanır)",
                  "BLS Polşa daxil hər şəkilli mətn captcha (avtomatik tapılır)",
                  "Любая текстовая капча-картинка, включая BLS Польша (находится автоматически)"),
    "desc_geetest": ("GeeTest v3/v4: slide, icon/word click, nine-grid, gobang",
                     "GeeTest v3/v4: kaydırma, simge/kelime tıklama, dokuzlu ızgara, gobang",
                     "GeeTest v3/v4: sürüşdürmə, ikon/söz kliki, doqquzlu şəbəkə, gobang",
                     "GeeTest v3/v4: слайдер, клик по иконкам/словам, сетка 3×3, гобанг"),
    "desc_recaptcha": ("Image challenge: 3×3, 4×4 and new tiles (“select all images with …”)",
                       "Resim testi: 3×3, 4×4 ve yeni kareler (“… olan tüm resimleri seçin”)",
                       "Şəkil testi: 3×3, 4×4 və yeni xanalar (“… olan bütün şəkilləri seçin”)",
                       "Задание с картинками: 3×3, 4×4 и новые клетки («Выберите все изображения, где есть …»)"),
    "closeBadge": ("Hide", "Gizle", "Gizlət", "Скрыть"),
    "solvingPlaceholder": ("Solving…", "Çözülüyor…", "Həll olunur…", "Решается…"),
    "allowAllSites": ("Allow on all sites", "Tüm sitelerde izin ver", "Bütün saytlarda icazə ver",
                      "Разрешить на всех сайтах"),
    "allowAllSitesHint": (
        "Needed for captcha pictures served from another domain: Captchosa then reads them from a screenshot of the visible tab.",
        "Başka bir alan adından gelen captcha resimleri için gerekli: Captchosa onları görünen sekmenin ekran görüntüsünden okur.",
        "Başqa domendən gələn captcha şəkilləri üçün lazımdır: Captchosa onları görünən tabın ekran görüntüsündən oxuyur.",
        "Нужно для картинок капчи с другого домена: Captchosa читает их со снимка видимой вкладки."),
    "allowAllSitesDone": ("Allowed on all sites", "Tüm sitelerde izinli", "Bütün saytlarda icazəlidir",
                          "Разрешено на всех сайтах"),
    "solvedCount": ("solved", "çözüldü", "həll", "решено"),
    "failedCount": ("failed", "hata", "xəta", "ошибок"),
    "never": ("never", "hiç", "heç vaxt", "никогда"),
    "behaviour": ("Behaviour", "Davranış", "Davranış", "Поведение"),
    "autoSolve": ("Auto-solve", "Otomatik çöz", "Avtomatik həll", "Автоматически"),
    "autoSolveHint": ("Off: a Solve button appears next to the captcha",
                      "Kapalı: captcha yanında Çöz düğmesi çıkar",
                      "Deaktiv: captcha yanında Həll et düyməsi çıxır",
                      "Выкл.: рядом с капчей появится кнопка «Решить»"),
    "delay": ("Delay before solving", "Çözmeden önce bekleme", "Həlldən əvvəl gözləmə", "Задержка перед решением"),
    "retries": ("Retries", "Tekrar sayısı", "Təkrar sayı", "Повторы"),
    "retriesHint": ("Rounds per captcha, e.g. after a wrong answer", "Captcha başına tur (yanlış cevaptan sonra)",
                    "Captcha başına raund (səhv cavabdan sonra)", "Раундов на капчу (после неверного ответа)"),
    "submitAfterFill": ("Submit after filling text", "Metni doldurunca gönder", "Mətni doldurduqdan sonra göndər",
                        "Отправлять после ввода текста"),
    "submitAfterFillHint": ("Text captchas only, and only with auto-solve", "Sadece metin captcha, otomatik çözmede",
                            "Yalnız mətn captcha, avtomatik həlldə", "Только текстовые капчи и автоматический режим"),
    "blsGridSubmit": ("Press Verify after BLS grid", "BLS ızgaradan sonra Verify'a bas", "BLS şəbəkədən sonra Verify bas",
                      "Нажимать Verify после сетки BLS"),
    "blsGridSubmitHint": ("Skipped while a password field is empty", "Şifre alanı boşsa atlanır",
                          "Şifrə sahəsi boşdursa ötürülür", "Пропускается, если пароль не введён"),
    "pageBadge": ("Status badge on the page", "Sayfada durum rozeti", "Səhifədə status nişanı", "Значок статуса на странице"),
    "pageBadgeHint": ("Small pill next to the captcha", "Captcha yanında küçük etiket", "Captcha yanında kiçik nişan",
                      "Небольшая метка рядом с капчей"),
    "language": ("Language", "Dil", "Dil", "Язык"),
    "languageAuto": ("Browser language", "Tarayıcı dili", "Brauzer dili", "Язык браузера"),
    "connection": ("Connection", "Bağlantı", "Bağlantı", "Подключение"),
    "serviceUrl": ("Service URL", "Servis adresi", "Xidmət ünvanı", "Адрес сервиса"),
    "serviceUrlHint": ("Your own Captchosa server, e.g. http://127.0.0.1:8790",
                       "Kendi Captchosa sunucunuz, ör. http://127.0.0.1:8790",
                       "Öz Captchosa serveriniz, məs. http://127.0.0.1:8790",
                       "Ваш сервер Captchosa, напр. http://127.0.0.1:8790"),
    "clientKey": ("Client key", "İstemci anahtarı", "Müştəri açarı", "Ключ клиента"),
    "show": ("Show", "Göster", "Göstər", "Показать"),
    "hide": ("Hide", "Gizle", "Gizlət", "Скрыть"),
    "save": ("Save", "Kaydet", "Yadda saxla", "Сохранить"),
    "test": ("Test connection", "Bağlantıyı test et", "Bağlantını yoxla", "Проверить связь"),
    "testing": ("Testing…", "Test ediliyor…", "Yoxlanılır…", "Проверка…"),
    "saved": ("Saved", "Kaydedildi", "Yadda saxlanıldı", "Сохранено"),
    "permDenied": ("Access to the service URL was not granted", "Servis adresine erişim izni verilmedi",
                   "Xidmət ünvanına icazə verilmədi", "Доступ к адресу сервиса не разрешён"),
    "badUrl": ("Enter a valid http(s) URL", "Geçerli bir http(s) adresi girin", "Düzgün http(s) ünvanı daxil edin",
               "Введите корректный http(s) адрес"),
    "connected": ("Connected", "Bağlı", "Qoşulub", "Подключено"),
    "offline": ("Offline", "Bağlantı yok", "Bağlantı yoxdur", "Нет связи"),
    "checking": ("Checking…", "Kontrol ediliyor…", "Yoxlanılır…", "Проверка…"),
    "notConfigured": ("Not configured", "Ayarlanmadı", "Qurulmayıb", "Не настроено"),
    "balance": ("Tokens", "Token", "Token", "Токены"),
    "unlimited": ("unlimited", "sınırsız", "limitsiz", "без лимита"),
    "version": ("Version", "Sürüm", "Versiya", "Версия"),
    "activity": ("Recent activity", "Son işlemler", "Son fəaliyyət", "Последние действия"),
    "clearLog": ("Clear", "Temizle", "Təmizlə", "Очистить"),
    "noActivity": ("Nothing solved yet", "Henüz çözüm yok", "Hələ heç nə həll olunmayıb", "Пока ничего не решено"),
    "openOptions": ("Settings", "Ayarlar", "Parametrlər", "Настройки"),
    "advanced": ("Advanced", "Gelişmiş", "Əlavə", "Дополнительно"),
    "resetStats": ("Reset counters", "Sayaçları sıfırla", "Sayğacları sıfırla", "Сбросить счётчики"),
    "statsReset": ("Counters reset", "Sayaçlar sıfırlandı", "Sayğaclar sıfırlandı", "Счётчики сброшены"),
    "noKey": ("Set the service URL and client key in Settings", "Servis adresini ve anahtarı Ayarlar'da girin",
              "Xidmət ünvanını və açarı Parametrlərdə daxil edin", "Укажите адрес сервиса и ключ в настройках"),
    "solveButton": ("Solve", "Çöz", "Həll et", "Решить"),
    "about": ("About", "Hakkında", "Haqqında", "О расширении"),
    "aboutText": ("Captchosa sends captcha images only to the service URL above. Nothing else leaves your browser.",
                  "Captchosa captcha resimlerini yalnızca yukarıdaki servis adresine gönderir. Başka hiçbir şey tarayıcıdan çıkmaz.",
                  "Captchosa captcha şəkillərini yalnız yuxarıdakı xidmət ünvanına göndərir. Başqa heç nə brauzerdən çıxmır.",
                  "Captchosa отправляет изображения капчи только на указанный выше адрес сервиса. Больше ничего не покидает браузер."),
    "ms": ("ms", "ms", "ms", "мс"),
}


def main():
    for i, lang in enumerate(LANGS):
        out = {k: {"message": v[i]} for k, v in STRINGS.items()}
        d = EXT / "_locales" / lang
        d.mkdir(parents=True, exist_ok=True)
        (d / "messages.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("locales:", ", ".join(LANGS))


if __name__ == "__main__":
    main()
