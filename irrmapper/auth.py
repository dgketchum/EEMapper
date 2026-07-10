import ee


def is_authorized(project='ee-dgketchum'):
    try:
        ee.Initialize(project=project)
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False
