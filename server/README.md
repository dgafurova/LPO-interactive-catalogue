# Orbital catalog

## Run Server

### Config

You need to fill /internal/app/config.py file

Required parameters:

| Name                         | Explanation                             | 
|------------------------------|-----------------------------------------|
| db_user                      | DB user name                            |
| db_password                  | DB user password                        |
| db_hostname                  | DB host (e.g. localhost)                |
| db_name                      | DB name (e.g. orbits)                   |
| db_port                      | DB port (e.g. 5432)                     |


### Run in docker
- Clone the repository
- Add dist directory (built frontend) to the root directory
- Build the docker image in the root directory of the project

```
docker build -t orbits:latest ./
```

- Run docker-compose `dev/docker-compose.yml`. It starts postgres & service docker containers

```
docker compose -f docker-compose.yml up -d
```

- Apply migration to your DB `./migration/1_init.up.sql`


### Run locally

- Run PostgreSQL (e.g. official docker image)
- Apply migration to your DB `./migration/1_init.up.sql`
- Start the service

```
make run-app
```

## More docs

All tables are stored in `./migration/1_init.up.sql`

[API docs](https://orbital-catalog.auditory.ru/docs)
