// Prova minima della cache KV mista nel meta backend: un tensore "cache_k_l0" f16 [1024, 512] allocato sul meta
// device (fette nel tipo di GGML_META_KV_TYPES), riempito con ggml_set_rows in due passate (righe 0-127, poi 128-129)
// come fanno due ubatch, poi riletto con ggml_backend_tensor_get e confrontato con l'ingresso.
//   test-meta-kv [n_devices]     (GGML_META_KV_TYPES=q8_0 oppure q8_0,f16)
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

static ggml_backend_meta_split_state split_cb(const ggml_tensor * t, void * ud) {
    const size_t n = *(const size_t *) ud;
    ggml_backend_meta_split_state s = {};
    s.n_segments = 1; s.nr[0] = 1;
    if (strcmp(t->name, "k_idxs") == 0 || strcmp(t->name, "x") == 0) {
        s.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        for (size_t j = 0; j < n; j++) s.ne[j] = t->ne[0];
        return s;
    }
    if (strcmp(t->name, "w") == 0) {
        s.axis = GGML_BACKEND_SPLIT_AXIS_1;
        for (size_t j = 0; j < n; j++) s.ne[j] = t->ne[1] / (int64_t) n;
        return s;
    }
    s.axis = GGML_BACKEND_SPLIT_AXIS_0;
    s.n_segments = 1;
    s.nr[0] = 1;
    for (size_t j = 0; j < n; j++) {
        s.ne[j] = t->ne[0] / (int64_t) n;
    }
    return s;
}

static float h2f(uint16_t h) {
    return ggml_fp16_to_fp32(h);
}

static ggml_type tipo_da_nome(const char * nome) {
    for (int i = 0; i < GGML_TYPE_COUNT; i++) {
        const char * n = ggml_type_name((ggml_type) i);
        if (n != nullptr && strcmp(n, nome) == 0) return (ggml_type) i;
    }
    printf("tipo sconosciuto: %s\n", nome); exit(1);
}

int main(int argc, char ** argv) {
    size_t n = argc > 1 ? (size_t) atoi(argv[1]) : 1;
    const int64_t ne0 = 1024, kv_size = 512;
    const int n_rows_total = getenv("NROWS") ? atoi(getenv("NROWS")) : 130;
    const int n_tok_fa     = getenv("NTOK")  ? atoi(getenv("NTOK"))  : 4;
    const ggml_type cache_type = getenv("KV_NATIVE") ? tipo_da_nome(getenv("KV_NATIVE")) : GGML_TYPE_F16;

    ggml_backend_load_all();
    std::vector<ggml_backend_dev_t> devs;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            devs.push_back(d);
        }
    }
    if (devs.size() < n) { printf("servono %zu GPU, trovate %zu\n", n, devs.size()); return 1; }
    for (size_t j = 0; j < n; j++) printf("dev %zu: %s\n", j, ggml_backend_dev_name(devs[j]));
    ggml_backend_dev_t meta = ggml_backend_meta_device(devs.data(), n, split_cb, &n);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(meta);
    ggml_backend_t backend = ggml_backend_dev_init(meta, nullptr);

    // cache statica
    ggml_init_params p0 = { ggml_tensor_overhead() * 8, nullptr, true };
    ggml_context * ctx0 = ggml_init(p0);
    ggml_tensor * k = ggml_new_tensor_2d(ctx0, cache_type, ne0, kv_size);
    ggml_set_name(k, "cache_k_l0");
    ggml_tensor * v = ggml_new_tensor_2d(ctx0, cache_type, ne0, kv_size);
    ggml_set_name(v, "cache_v_l0");
    const int64_t n_in = 64;
    ggml_tensor * w = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_in, ne0);
    ggml_set_name(w, "w");
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx0, buft);
    ggml_backend_buffer_clear(buf, 0);
    std::vector<float> wv((size_t) n_in * ne0);
    for (size_t i = 0; i < wv.size(); i++) wv[i] = sinf(0.01f * (float) i) * (0.5f + (float) ((i * 7919) % 13) / 13.0f);
    ggml_backend_tensor_set(w, wv.data(), 0, wv.size() * sizeof(float));
    printf("cache allocata: %zu byte (meta)\n", ggml_backend_buffer_get_size(buf));

    // ingressi deterministici
    // atteso: riga t = colonna (t % n_in) di w, scalata per (1 + t/100): k_cur[i, t] = w[t % n_in, i] * s_t
    std::vector<float> all((size_t) ne0 * n_rows_total);
    for (int t = 0; t < n_rows_total; t++) {
        const float st = 1.0f + (float) t / 100.0f;
        for (int64_t i = 0; i < ne0; i++) all[(size_t) t * ne0 + i] = wv[(size_t) i * n_in + (t % n_in)] * st;
    }

    const int n_pass = getenv("ONEPASS") ? 1 : 2;
    const int passi[2][2] = { {0, n_pass == 1 ? n_rows_total : 128}, {128, n_rows_total} };
    for (int pass = 0; pass < n_pass; pass++) {
        const int r0 = passi[pass][0], r1 = passi[pass][1], nt = r1 - r0;
        ggml_init_params pg = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctxg = ggml_init(pg);
        ggml_tensor * x = ggml_new_tensor_2d(ctxg, GGML_TYPE_F32, n_in, nt);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * k_cur = ggml_mul_mat(ctxg, w, x);
        ggml_set_name(k_cur, "k_cur");
        ggml_tensor * idxs = ggml_new_tensor_1d(ctxg, GGML_TYPE_I64, nt);
        ggml_set_name(idxs, "k_idxs"); ggml_set_input(idxs);
        ggml_tensor * kv = ggml_view_2d(ctxg, k, ne0, kv_size, k->nb[1], 0);
        ggml_tensor * res = ggml_set_rows(ctxg, kv, k_cur, idxs);
        ggml_set_name(res, "k_set"); ggml_set_output(res);
        ggml_tensor * vv = ggml_view_2d(ctxg, v, ne0, kv_size, v->nb[1], 0);
        ggml_tensor * resv = ggml_set_rows(ctxg, vv, k_cur, idxs);
        ggml_set_name(resv, "v_set"); ggml_set_output(resv);
        ggml_cgraph * gf = ggml_new_graph(ctxg);
        ggml_build_forward_expand(gf, res);
        ggml_build_forward_expand(gf, resv);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("alloc del grafo fallita\n"); return 1; }
        std::vector<float> xv((size_t) n_in * nt, 0.0f);
        for (int t = 0; t < nt; t++) xv[(size_t) t * n_in + ((r0 + t) % n_in)] = 1.0f + (float) (r0 + t) / 100.0f;
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        std::vector<int64_t> id(nt);
        for (int i = 0; i < nt; i++) id[i] = r0 + i;
        ggml_backend_tensor_set(idxs, id.data(), 0, nt * sizeof(int64_t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute fallito\n"); return 1; }
        ggml_backend_synchronize(backend);
        printf("passata %d: righe %d-%d scritte\n", pass, r0, r1 - 1);
        ggml_gallocr_free(ga);
        ggml_free(ctxg);
    }

    // rilettura (passa per la conversione del meta backend se la fetta e' di un altro tipo)
    std::vector<uint16_t> out((size_t) ne0 * n_rows_total);
    if (cache_type == GGML_TYPE_F16) {
        ggml_backend_tensor_get(k, out.data(), 0, out.size() * sizeof(uint16_t));
    } else {
        std::vector<char> raw(ggml_row_size(cache_type, ne0) * n_rows_total);
        ggml_backend_tensor_get(k, raw.data(), 0, raw.size());
        std::vector<float> tmp((size_t) ne0 * n_rows_total);
        ggml_get_type_traits(cache_type)->to_float(raw.data(), tmp.data(), tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) out[i] = ggml_fp32_to_fp16(tmp[i]);
    }
    int bad = 0;
    for (int r = 0; r < n_rows_total; r++) {
        double num = 0, den = 0; int nnan = 0;
        for (int64_t i = 0; i < ne0; i++) {
            const float a = h2f(out[(size_t) r * ne0 + i]), b = all[(size_t) r * ne0 + i];
            if (a != a) nnan++;
            num += (double) (a - b) * (a - b); den += (double) b * b;
        }
        const double rel = sqrt(num / (den + 1e-12));
        const bool ok = rel < 0.02 && nnan == 0;
        if (!ok) bad++;
        if (r < 2 || r == 127 || r >= 128 || !ok) {
            printf("riga %3d: errore relativo %.4f  nan=%d  %s\n", r, rel, nnan, ok ? "ok" : "SBAGLIATA");
        }
    }
    printf("%s: %d righe sbagliate su %d\n", bad == 0 ? "OK" : "FALLITO", bad, n_rows_total);

    // flash attention: 4 query (posizioni 126..129, mascherate causali) contro n_kv = 256, letta via viste permutate
    {
        const int64_t hd = 128, nh = ne0 / hd, n_tok = n_tok_fa, n_kv = GGML_PAD(n_rows_total, 256);
        ggml_init_params pg = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctxg = ggml_init(pg);
        ggml_tensor * x = ggml_new_tensor_2d(ctxg, GGML_TYPE_F32, n_in, n_tok);
        ggml_set_name(x, "x"); ggml_set_input(x);
        ggml_tensor * q2 = ggml_mul_mat(ctxg, w, x);                       // [1024, n_tok], split asse 0
        ggml_tensor * q3 = ggml_reshape_3d(ctxg, q2, hd, nh, n_tok);
        ggml_tensor * q  = ggml_permute(ctxg, q3, 0, 2, 1, 3);             // [hd, n_tok, nh]
        ggml_tensor * kk = ggml_permute(ctxg, ggml_view_3d(ctxg, k, hd, nh, n_kv, ggml_row_size(k->type, hd), ggml_row_size(k->type, ne0), 0), 0, 2, 1, 3);
        ggml_tensor * vk = ggml_permute(ctxg, ggml_view_3d(ctxg, v, hd, nh, n_kv, ggml_row_size(v->type, hd), ggml_row_size(v->type, ne0), 0), 0, 2, 1, 3);
        ggml_tensor * mask = ggml_new_tensor_2d(ctxg, GGML_TYPE_F16, n_kv, n_tok);
        ggml_set_name(mask, "kq_mask"); ggml_set_input(mask);
        ggml_tensor * fa = ggml_flash_attn_ext(ctxg, q, kk, vk, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
        ggml_set_name(fa, "fa"); ggml_set_output(fa);
        ggml_cgraph * gf = ggml_new_graph(ctxg);
        ggml_build_forward_expand(gf, fa);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        if (!ggml_gallocr_alloc_graph(ga, gf)) { printf("alloc FA fallita\n"); return 1; }
        std::vector<float> xv((size_t) n_in * n_tok, 0.0f);
        const int pos0 = n_rows_total - (int) n_tok;   // le query sono gli ultimi n_tok token
        for (int t = 0; t < n_tok; t++) xv[(size_t) t * n_in + ((pos0 + t) % n_in)] = 1.0f;
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size() * sizeof(float));
        std::vector<uint16_t> mv((size_t) mask->ne[0] * mask->ne[1], ggml_fp32_to_fp16(-INFINITY));
        for (int t = 0; t < n_tok; t++) for (int64_t j = 0; j <= pos0 + t; j++) mv[(size_t) t * n_kv + j] = ggml_fp32_to_fp16(0.0f);
        ggml_backend_tensor_set(mask, mv.data(), 0, mv.size() * sizeof(uint16_t));
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) { printf("compute FA fallito\n"); return 1; }
        ggml_backend_synchronize(backend);
        std::vector<float> outf((size_t) ggml_nelements(fa));
        ggml_backend_tensor_get(fa, outf.data(), 0, outf.size() * sizeof(float));
        const char * fn = getenv("FA_OUT") ? getenv("FA_OUT") : "fa-out.bin";
        FILE * f = fopen(fn, "wb"); fwrite(outf.data(), sizeof(float), outf.size(), f); fclose(f);
        double ss = 0; int nn = 0; for (float y : outf) { if (y != y) nn++; else ss += (double) y * y; }
        printf("FA: %zu valori, rms %.4f, nan=%d, salvato in %s\n", outf.size(), sqrt(ss / outf.size()), nn, fn);
        ggml_gallocr_free(ga); ggml_free(ctxg);
    }
    return bad == 0 ? 0 : 1;
}
