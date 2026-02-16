#include "pkcs11_signer.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Minimal PKCS#11 type set needed for C_Initialize/C_Sign with CKM_EDDSA. */
typedef unsigned char CK_BYTE;
typedef unsigned char CK_BBOOL;
typedef unsigned long CK_ULONG;
typedef CK_ULONG CK_RV;
typedef CK_ULONG CK_SLOT_ID;
typedef CK_SLOT_ID *CK_SLOT_ID_PTR;
typedef CK_ULONG CK_SESSION_HANDLE;
typedef CK_ULONG CK_OBJECT_HANDLE;
typedef CK_ULONG CK_FLAGS;
typedef CK_ULONG CK_MECHANISM_TYPE;
typedef CK_BYTE *CK_BYTE_PTR;
typedef CK_ULONG *CK_ULONG_PTR;
typedef void *CK_VOID_PTR;
typedef CK_BYTE CK_UTF8CHAR;
typedef CK_UTF8CHAR *CK_UTF8CHAR_PTR;
typedef CK_SESSION_HANDLE *CK_SESSION_HANDLE_PTR;
typedef CK_OBJECT_HANDLE *CK_OBJECT_HANDLE_PTR;
typedef CK_RV (*CK_NOTIFY)(CK_SESSION_HANDLE, CK_ULONG, CK_VOID_PTR);

typedef struct CK_MECHANISM {
    CK_MECHANISM_TYPE mechanism;
    CK_VOID_PTR pParameter;
    CK_ULONG ulParameterLen;
} CK_MECHANISM;
typedef CK_MECHANISM *CK_MECHANISM_PTR;

typedef struct CK_ATTRIBUTE {
    CK_ULONG type;
    CK_VOID_PTR pValue;
    CK_ULONG ulValueLen;
} CK_ATTRIBUTE;
typedef CK_ATTRIBUTE *CK_ATTRIBUTE_PTR;

typedef CK_RV (*C_Initialize_t)(CK_VOID_PTR pInitArgs);
typedef CK_RV (*C_Finalize_t)(CK_VOID_PTR pReserved);
typedef CK_RV (*C_GetSlotList_t)(CK_BBOOL tokenPresent, CK_SLOT_ID_PTR pSlotList, CK_ULONG_PTR pulCount);
typedef CK_RV (*C_OpenSession_t)(CK_SLOT_ID slotID, CK_FLAGS flags, CK_VOID_PTR pApplication, CK_NOTIFY Notify, CK_SESSION_HANDLE_PTR phSession);
typedef CK_RV (*C_CloseSession_t)(CK_SESSION_HANDLE hSession);
typedef CK_RV (*C_Login_t)(CK_SESSION_HANDLE hSession, CK_ULONG userType, CK_UTF8CHAR_PTR pPin, CK_ULONG ulPinLen);
typedef CK_RV (*C_Logout_t)(CK_SESSION_HANDLE hSession);
typedef CK_RV (*C_FindObjectsInit_t)(CK_SESSION_HANDLE hSession, CK_ATTRIBUTE_PTR pTemplate, CK_ULONG ulCount);
typedef CK_RV (*C_FindObjects_t)(CK_SESSION_HANDLE hSession, CK_OBJECT_HANDLE_PTR phObject, CK_ULONG ulMaxObjectCount, CK_ULONG_PTR pulObjectCount);
typedef CK_RV (*C_FindObjectsFinal_t)(CK_SESSION_HANDLE hSession);
typedef CK_RV (*C_SignInit_t)(CK_SESSION_HANDLE hSession, CK_MECHANISM_PTR pMechanism, CK_OBJECT_HANDLE hKey);
typedef CK_RV (*C_Sign_t)(CK_SESSION_HANDLE hSession, CK_BYTE_PTR pData, CK_ULONG ulDataLen, CK_BYTE_PTR pSignature, CK_ULONG_PTR pulSignatureLen);

#define CKR_OK 0UL
#define CKF_SERIAL_SESSION 0x00000004UL
#define CKF_RW_SESSION 0x00000002UL
#define CKU_USER 1UL
#define CKO_PRIVATE_KEY 0x00000003UL
#define CKA_CLASS 0x00000000UL
#define CKA_LABEL 0x00000003UL
#define CKA_ID 0x00000102UL
#define CKM_EDDSA 0x0000108AUL

struct Pkcs11State {
    void *module;
    CK_SLOT_ID slot_id;
    CK_SESSION_HANDLE session;
    CK_OBJECT_HANDLE key;
    unsigned char *last_signature;
    size_t last_signature_len;

    C_Initialize_t C_Initialize;
    C_Finalize_t C_Finalize;
    C_GetSlotList_t C_GetSlotList;
    C_OpenSession_t C_OpenSession;
    C_CloseSession_t C_CloseSession;
    C_Login_t C_Login;
    C_Logout_t C_Logout;
    C_FindObjectsInit_t C_FindObjectsInit;
    C_FindObjects_t C_FindObjects;
    C_FindObjectsFinal_t C_FindObjectsFinal;
    C_SignInit_t C_SignInit;
    C_Sign_t C_Sign;
};

static struct Pkcs11State g_pkcs11 = {0};

static int load_symbol(void **target, const char *name) {
    *target = dlsym(g_pkcs11.module, name);
    if (*target == NULL) {
        fprintf(stderr, "Missing PKCS#11 symbol: %s\n", name);
        return -1;
    }
    return 0;
}

static int hex_to_bytes(const char *hex, unsigned char *out, size_t *out_len) {
    size_t hex_len = strlen(hex);
    if ((hex_len % 2) != 0 || *out_len < hex_len / 2) {
        return -1;
    }

    for (size_t i = 0; i < hex_len; i += 2) {
        unsigned int value = 0;
        if (sscanf(hex + i, "%2x", &value) != 1) {
            return -1;
        }
        out[i / 2] = (unsigned char)value;
    }

    *out_len = hex_len / 2;
    return 0;
}

int pkcs11_initialize_from_env(void) {
    const char *module_path = getenv("PKCS11_MODULE_PATH");
    const char *pin = getenv("PKCS11_PIN");
    const char *label = getenv("HSM_KEY_LABEL");
    const char *id_hex = getenv("HSM_KEY_ID_HEX");

    if (module_path == NULL || pin == NULL || (label == NULL && id_hex == NULL)) {
        fprintf(stderr, "PKCS#11 env missing. Set PKCS11_MODULE_PATH, PKCS11_PIN, and HSM_KEY_LABEL or HSM_KEY_ID_HEX.\n");
        return -1;
    }

    g_pkcs11.module = dlopen(module_path, RTLD_NOW | RTLD_LOCAL);
    if (g_pkcs11.module == NULL) {
        fprintf(stderr, "dlopen failed for %s: %s\n", module_path, dlerror());
        return -1;
    }

    if (load_symbol((void **)&g_pkcs11.C_Initialize, "C_Initialize") != 0 ||
        load_symbol((void **)&g_pkcs11.C_Finalize, "C_Finalize") != 0 ||
        load_symbol((void **)&g_pkcs11.C_GetSlotList, "C_GetSlotList") != 0 ||
        load_symbol((void **)&g_pkcs11.C_OpenSession, "C_OpenSession") != 0 ||
        load_symbol((void **)&g_pkcs11.C_CloseSession, "C_CloseSession") != 0 ||
        load_symbol((void **)&g_pkcs11.C_Login, "C_Login") != 0 ||
        load_symbol((void **)&g_pkcs11.C_Logout, "C_Logout") != 0 ||
        load_symbol((void **)&g_pkcs11.C_FindObjectsInit, "C_FindObjectsInit") != 0 ||
        load_symbol((void **)&g_pkcs11.C_FindObjects, "C_FindObjects") != 0 ||
        load_symbol((void **)&g_pkcs11.C_FindObjectsFinal, "C_FindObjectsFinal") != 0 ||
        load_symbol((void **)&g_pkcs11.C_SignInit, "C_SignInit") != 0 ||
        load_symbol((void **)&g_pkcs11.C_Sign, "C_Sign") != 0) {
        dlclose(g_pkcs11.module);
        memset(&g_pkcs11, 0, sizeof(g_pkcs11));
        return -1;
    }

    CK_RV rv = g_pkcs11.C_Initialize(NULL);
    if (rv != CKR_OK) {
        fprintf(stderr, "C_Initialize failed: 0x%lx\n", rv);
        pkcs11_cleanup();
        return -1;
    }

    CK_ULONG slot_count = 0;
    rv = g_pkcs11.C_GetSlotList(1, NULL, &slot_count);
    if (rv != CKR_OK || slot_count == 0) {
        fprintf(stderr, "C_GetSlotList failed or no slots: rv=0x%lx count=%lu\n", rv, slot_count);
        pkcs11_cleanup();
        return -1;
    }

    CK_SLOT_ID *slots = calloc(slot_count, sizeof(CK_SLOT_ID));
    if (slots == NULL) {
        fprintf(stderr, "Out of memory loading slot list.\n");
        pkcs11_cleanup();
        return -1;
    }

    rv = g_pkcs11.C_GetSlotList(1, slots, &slot_count);
    if (rv != CKR_OK) {
        fprintf(stderr, "C_GetSlotList(list) failed: 0x%lx\n", rv);
        free(slots);
        pkcs11_cleanup();
        return -1;
    }

    g_pkcs11.slot_id = slots[0];
    free(slots);

    rv = g_pkcs11.C_OpenSession(g_pkcs11.slot_id, CKF_SERIAL_SESSION | CKF_RW_SESSION, NULL, NULL, &g_pkcs11.session);
    if (rv != CKR_OK) {
        fprintf(stderr, "C_OpenSession failed: 0x%lx\n", rv);
        pkcs11_cleanup();
        return -1;
    }

    rv = g_pkcs11.C_Login(g_pkcs11.session, CKU_USER, (CK_UTF8CHAR_PTR)pin, (CK_ULONG)strlen(pin));
    if (rv != CKR_OK) {
        fprintf(stderr, "C_Login failed: 0x%lx\n", rv);
        pkcs11_cleanup();
        return -1;
    }

    CK_ULONG key_class = CKO_PRIVATE_KEY;
    CK_ATTRIBUTE attrs[3];
    CK_ULONG attr_count = 0;
    attrs[attr_count++] = (CK_ATTRIBUTE){CKA_CLASS, &key_class, sizeof(key_class)};

    unsigned char id_bytes[64] = {0};
    size_t id_len = sizeof(id_bytes);
    if (id_hex != NULL) {
        if (hex_to_bytes(id_hex, id_bytes, &id_len) != 0) {
            fprintf(stderr, "Invalid HSM_KEY_ID_HEX value.\n");
            pkcs11_cleanup();
            return -1;
        }
        attrs[attr_count++] = (CK_ATTRIBUTE){CKA_ID, id_bytes, (CK_ULONG)id_len};
    } else {
        attrs[attr_count++] = (CK_ATTRIBUTE){CKA_LABEL, (void *)label, (CK_ULONG)strlen(label)};
    }

    rv = g_pkcs11.C_FindObjectsInit(g_pkcs11.session, attrs, attr_count);
    if (rv != CKR_OK) {
        fprintf(stderr, "C_FindObjectsInit failed: 0x%lx\n", rv);
        pkcs11_cleanup();
        return -1;
    }

    CK_ULONG found = 0;
    rv = g_pkcs11.C_FindObjects(g_pkcs11.session, &g_pkcs11.key, 1, &found);
    g_pkcs11.C_FindObjectsFinal(g_pkcs11.session);
    if (rv != CKR_OK || found == 0) {
        fprintf(stderr, "No key handle found in token (rv=0x%lx found=%lu).\n", rv, found);
        pkcs11_cleanup();
        return -1;
    }

    return 0;
}

int sign_log_hsm(unsigned char *data, size_t len) {
    CK_MECHANISM mech = {CKM_EDDSA, NULL, 0};
    CK_RV rv = g_pkcs11.C_SignInit(g_pkcs11.session, &mech, g_pkcs11.key);
    if (rv != CKR_OK) {
        fprintf(stderr, "C_SignInit failed: 0x%lx\n", rv);
        return -1;
    }

    CK_ULONG sig_len = 0;
    rv = g_pkcs11.C_Sign(g_pkcs11.session, data, (CK_ULONG)len, NULL, &sig_len);
    if (rv != CKR_OK || sig_len == 0) {
        fprintf(stderr, "C_Sign(length) failed: 0x%lx\n", rv);
        return -1;
    }

    unsigned char *sig = calloc(sig_len, 1);
    if (sig == NULL) {
        fprintf(stderr, "Out of memory allocating signature buffer.\n");
        return -1;
    }

    rv = g_pkcs11.C_Sign(g_pkcs11.session, data, (CK_ULONG)len, sig, &sig_len);
    if (rv != CKR_OK) {
        fprintf(stderr, "C_Sign failed: 0x%lx\n", rv);
        free(sig);
        return -1;
    }

    free(g_pkcs11.last_signature);
    g_pkcs11.last_signature = sig;
    g_pkcs11.last_signature_len = sig_len;
    return 0;
}

const unsigned char *pkcs11_last_signature(size_t *len_out) {
    if (len_out != NULL) {
        *len_out = g_pkcs11.last_signature_len;
    }
    return g_pkcs11.last_signature;
}

void pkcs11_cleanup(void) {
    if (g_pkcs11.last_signature != NULL) {
        free(g_pkcs11.last_signature);
    }

    if (g_pkcs11.C_Logout != NULL && g_pkcs11.session != 0) {
        g_pkcs11.C_Logout(g_pkcs11.session);
    }

    if (g_pkcs11.C_CloseSession != NULL && g_pkcs11.session != 0) {
        g_pkcs11.C_CloseSession(g_pkcs11.session);
    }

    if (g_pkcs11.C_Finalize != NULL) {
        g_pkcs11.C_Finalize(NULL);
    }

    if (g_pkcs11.module != NULL) {
        dlclose(g_pkcs11.module);
    }

    memset(&g_pkcs11, 0, sizeof(g_pkcs11));
}
