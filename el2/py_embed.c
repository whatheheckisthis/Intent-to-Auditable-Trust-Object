#include <Python.h>
#include "py_embed.h"
#include "uart.h"

static PyObject *iato_py_validator = NULL;
static PyObject *iato_py_validate_method = NULL;
static int iato_py_initialised = 0;

static void iato_uart_pyerr(const char *prefix) {
    iato_uart_puts(prefix);
    iato_uart_puts("\n");
}

int iato_py_embed_init(void) {
    PyObject *module;
    PyObject *klass;
    if (iato_py_initialised != 0) {
        return IATO_PY_OK;
    }
    Py_NoSiteFlag = 1;
    Py_IgnoreEnvironmentFlag = 0;
    Py_InitializeEx(0);
    if (!Py_IsInitialized()) {
        return IATO_PY_ERR_INIT;
    }
    PyRun_SimpleString("import sys; sys.path.insert(0, '/opt/iato')");
    module = PyImport_ImportModule("src.el2_validator");
    if (module == NULL) {
        PyErr_Clear();
        return IATO_PY_ERR_IMPORT;
    }
    klass = PyObject_GetAttrString(module, "El2CredentialValidator");
    if ((klass == NULL) || !PyCallable_Check(klass)) {
        Py_XDECREF(klass);
        Py_DECREF(module);
        return IATO_PY_ERR_IMPORT;
    }
    iato_py_validator = PyObject_CallObject(klass, NULL);
    Py_DECREF(klass);
    Py_DECREF(module);
    if (iato_py_validator == NULL) {
        PyErr_Clear();
        return IATO_PY_ERR_IMPORT;
    }
    iato_py_validate_method = PyObject_GetAttrString(iato_py_validator, "validate");
    if ((iato_py_validate_method == NULL) || !PyCallable_Check(iato_py_validate_method)) {
        Py_XDECREF(iato_py_validate_method);
        Py_DECREF(iato_py_validator);
        iato_py_validator = NULL;
        return IATO_PY_ERR_IMPORT;
    }
    iato_py_initialised = 1;
    return IATO_PY_OK;
}

int iato_py_validate_credential(const uint8_t *cred, size_t cred_len, uint32_t stream_id,
                                uint64_t *out_pa_base, uint64_t *out_pa_limit, uint8_t *out_permissions) {
    PyGILState_STATE g;
    PyObject *bytes;
    PyObject *args;
    PyObject *kwargs;
    PyObject *ret;
    PyObject *v;
    if ((iato_py_initialised == 0) || (iato_py_validate_method == NULL)) {
        return IATO_PY_ERR_INIT;
    }
    g = PyGILState_Ensure();
    bytes = PyBytes_FromStringAndSize((const char *)cred, (Py_ssize_t)cred_len);
    args = PyTuple_New(0);
    kwargs = PyDict_New();
    PyDict_SetItemString(kwargs, "raw", bytes);
    PyDict_SetItemString(kwargs, "stream_id", PyLong_FromUnsignedLong((unsigned long)stream_id));
    ret = PyObject_Call(iato_py_validate_method, args, kwargs);
    Py_DECREF(bytes);
    Py_DECREF(args);
    Py_DECREF(kwargs);
    if (ret == NULL) {
        if (PyErr_ExceptionMatches(PyExc_RuntimeError)) {
            iato_uart_pyerr("[iato][py] runtime error");
        } else {
            iato_uart_pyerr("[iato][py] validate failed");
        }
        PyErr_Clear();
        PyGILState_Release(g);
        return IATO_PY_ERR_CALL;
    }
    v = PyObject_GetAttrString(ret, "pa_range_base");
    if (v == NULL) { Py_DECREF(ret); PyGILState_Release(g); return IATO_PY_ERR_RESULT; }
    *out_pa_base = (uint64_t)PyLong_AsUnsignedLongLong(v);
    Py_DECREF(v);
    v = PyObject_GetAttrString(ret, "pa_range_limit");
    if (v == NULL) { Py_DECREF(ret); PyGILState_Release(g); return IATO_PY_ERR_RESULT; }
    *out_pa_limit = (uint64_t)PyLong_AsUnsignedLongLong(v);
    Py_DECREF(v);
    v = PyObject_GetAttrString(ret, "permissions");
    if (v == NULL) { Py_DECREF(ret); PyGILState_Release(g); return IATO_PY_ERR_RESULT; }
    *out_permissions = (uint8_t)PyLong_AsUnsignedLong(v);
    Py_DECREF(v);
    Py_DECREF(ret);
    PyGILState_Release(g);
    return IATO_PY_OK;
}

void iato_py_embed_shutdown(void) {
    if (iato_py_initialised == 0) {
        return;
    }
    Py_XDECREF(iato_py_validate_method);
    Py_XDECREF(iato_py_validator);
    iato_py_validate_method = NULL;
    iato_py_validator = NULL;
    Py_FinalizeEx();
    iato_py_initialised = 0;
}
