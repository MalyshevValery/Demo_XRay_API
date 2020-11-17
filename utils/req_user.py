import requests


def get_user_info(cookie, kratos_url):
    """Method to get info about user from kratos

    :param cookie: Ory session cookie
    :param kratos_url: URL of Public Kratos API
    :return: dict with all information about user
    """
    r = requests.get(
        f'{kratos_url}/sessions/whoami',
        cookies={'ory_kratos_session': cookie})
    return r.json()