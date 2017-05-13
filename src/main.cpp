#include <Python.h>
#include <cstdio>
#include <wchar.h>

int main(int argc, char **argv) {
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);

    if (program == NULL)
        exit(1);

    Py_SetProgramName(program);
    Py_Initialize();

    PyRun_SimpleString("exec(open('../src/app.py').read()); main()");

    Py_Finalize();

    PyMem_RawFree(program);

    return 0;
}
