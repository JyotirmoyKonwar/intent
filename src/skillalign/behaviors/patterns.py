"""Tables shared by the Python, shell, JavaScript and config detectors."""

from __future__ import annotations

import re
from urllib.parse import urlparse

#: Paths that hold secrets. A read is CREDENTIAL_ACCESS.
CREDENTIAL_PATHS = (
    ".aws/credentials", ".aws/config", ".ssh/", "id_rsa", "id_ed25519", "id_ecdsa",
    ".netrc", ".npmrc", ".pypirc", ".docker/config.json", ".kube/config", ".git-credentials",
    ".gnupg", ".config/gcloud", ".azure/", "/etc/shadow", "keychain", "login.keychain",
    "cookies.sqlite", "login data", "wallet.dat", ".env", "secrets.json", "credentials.json",
    "token.json", "service-account", ".config/gh/hosts.yml", ".terraform.d/credentials",
)

#: Paths that are private but not credentials. A read is FS_READ, and the data is a
#: sensitive source for exfiltration.
SENSITIVE_PATHS = (".gitconfig", ".bash_history", ".zsh_history", "/etc/passwd", ".viminfo",
                   ".python_history", "known_hosts", "/proc/", ".mozilla", ".config/google-chrome")

#: Writing here keeps code running later.
PERSISTENCE_PATHS = (
    ".bashrc", ".zshrc", ".profile", ".bash_profile", ".zprofile", ".zshenv", ".bash_login",
    "/etc/cron", "/var/spool/cron", "launchagents", "launchdaemons", "/etc/systemd",
    ".config/systemd", "authorized_keys", ".config/autostart", "/etc/rc.local", ".git/hooks/",
    "/etc/profile", "startup", "/etc/init.d",
)

#: Writing here changes the machine or the user's tools.
SYSCONFIG_PATHS = ("/etc/", ".gitconfig", ".config/", ".npmrc", ".condarc", "/usr/local/",
                   "hosts", ".pip/", "pip.conf")

SECRET_NAME = re.compile(r"(key|token|secret|passw|credential|auth|cookie|session)", re.I)


def path_class(path: str | None) -> str | None:
    """"persistence", "credential", "sensitive", "sysconfig" or None."""
    if not path:
        return None
    p = path.lower().replace("\\", "/")
    for group, name in ((PERSISTENCE_PATHS, "persistence"), (CREDENTIAL_PATHS, "credential"),
                        (SENSITIVE_PATHS, "sensitive")):
        if any(token in p for token in group):
            return name
    return None


def is_sysconfig(path: str | None) -> bool:
    p = (path or "").lower()
    return any(p.startswith(x) or f"/{x.lstrip('/')}" in p for x in SYSCONFIG_PATHS)


def host_of(url: str | None) -> str | None:
    """The host of an http, ftp, s3 or ws URL, or None."""
    if not url:
        return None
    m = re.search(r"(?:https?|ftp|wss?|s3|gs)://[^\s'\"`<>)]+", url)
    if not m:
        return None
    parsed = urlparse(m.group(0))
    host = parsed.hostname or parsed.netloc or None
    if host and "{" not in host and "$" not in host:
        return host if parsed.scheme not in ("s3", "gs") else f"{parsed.scheme}://{host}"
    return None


# ---------------------------------------------------------------- Python call tables

NET_CALLS = {
    "requests.get", "requests.post", "requests.put", "requests.patch", "requests.delete",
    "requests.head", "requests.request", "requests.Session", "httpx.get", "httpx.post",
    "httpx.put", "httpx.patch", "httpx.delete", "httpx.request", "httpx.stream",
    "httpx.Client", "httpx.AsyncClient", "urllib.request.urlopen", "urllib.request.urlretrieve",
    "urllib.urlopen", "urllib3.request", "urllib3.PoolManager", "http.client.HTTPSConnection",
    "http.client.HTTPConnection", "aiohttp.ClientSession", "socket.create_connection",
    "websocket.create_connection", "websockets.connect", "ftplib.FTP", "smtplib.SMTP",
    "smtplib.SMTP_SSL", "telnetlib.Telnet", "paramiko.SSHClient", "pycurl.Curl",
}
#: Method names that send data when called on a client object.
NET_METHODS = {"upload_file", "upload_fileobj", "put_object", "sendall", "sendto", "send_message",
               "sendmail", "send_mail", "post", "put", "patch"}
#: A call returns data only when it is a read.
NET_DATA_KW = {"data", "json", "files", "content", "body", "payload"}

PROC_CALLS = {
    "subprocess.run", "subprocess.call", "subprocess.check_call", "subprocess.check_output",
    "subprocess.Popen", "subprocess.getoutput", "subprocess.getstatusoutput", "os.system",
    "os.popen", "os.execv", "os.execve", "os.execvp", "os.execl", "os.execlp", "os.spawnl",
    "os.spawnv", "pty.spawn", "commands.getoutput", "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell", "sh.Command",
}
EVAL_CALLS = {"eval", "exec", "compile", "pickle.loads", "pickle.load", "marshal.loads",
              "cloudpickle.loads", "dill.loads", "importlib.import_module", "__import__",
              "runpy.run_path", "runpy.run_module", "imp.load_source"}
ENCODE_CALLS = {"base64.b64encode", "base64.b64decode", "base64.urlsafe_b64encode",
                "base64.urlsafe_b64decode", "base64.encodebytes", "base64.decodebytes",
                "base64.b32encode", "base64.b85encode", "base64.a85encode", "binascii.hexlify",
                "binascii.unhexlify", "binascii.b2a_base64", "binascii.a2b_base64",
                "codecs.encode", "codecs.decode", "zlib.compress", "zlib.decompress"}
ENV_CALLS = {"os.getenv", "os.environ.get", "os.environ.copy", "os.environ.items",
             "os.environ.keys", "os.environ.values", "os.environb.get", "dotenv.load_dotenv",
             "dotenv.dotenv_values"}
DELETE_CALLS = {"os.remove", "os.unlink", "os.rmdir", "os.removedirs", "shutil.rmtree",
                "send2trash.send2trash"}
ENUM_CALLS = {"os.listdir", "os.walk", "os.scandir", "glob.glob", "glob.iglob", "fnmatch.filter"}
WRITE_CALLS = {"shutil.copy", "shutil.copy2", "shutil.copyfile", "shutil.copytree",
               "shutil.move", "os.rename", "os.replace", "os.makedirs", "os.mkdir", "os.symlink",
               "os.link", "os.chmod", "os.chown", "tempfile.mkstemp", "json.dump", "pickle.dump",
               "yaml.dump", "yaml.safe_dump", "csv.writer", "shutil.make_archive"}
PKG_CALLS = {"pip.main", "pip._internal.main", "setuptools.setup", "subprocess.pip"}
INTERACT_CALLS = {"input", "getpass.getpass", "click.prompt", "click.confirm"}
PATH_READ_METHODS = {"read_text", "read_bytes"}
PATH_WRITE_METHODS = {"write_text", "write_bytes", "touch", "mkdir"}
PATH_DELETE_METHODS = {"unlink", "rmdir"}
PATH_ENUM_METHODS = {"glob", "rglob", "iterdir"}

#: Library calls (by last name) that read the file named by their first argument.
READ_SUFFIX = {"PdfReader", "load_workbook", "imread", "read_csv", "read_json", "read_excel",
               "read_parquet", "read_table", "read_feather", "loadtxt", "genfromtxt",
               "read_pickle", "read_xml", "read_html", "read_sql", "VideoCapture"}
READ_FULL = {"PIL.Image.open", "Image.open", "fitz.open", "docx.Document", "np.load",
             "numpy.load", "cv2.imread", "soundfile.read", "librosa.load", "wave.open",
             "zipfile.ZipFile", "tarfile.open", "sqlite3.connect", "openpyxl.load_workbook"}
#: Constructors whose objects later `write`/`save` to a path.
WRITER_FACTORIES = {"PdfWriter", "PdfFileWriter", "Workbook", "Document", "ZipFile"}
#: Methods that write to the path in their first argument.
WRITE_METHODS = {"to_csv", "to_json", "to_excel", "to_parquet", "to_pickle", "to_feather",
                 "savefig", "imwrite", "to_html", "to_xml", "to_markdown"}
