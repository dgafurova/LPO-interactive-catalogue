CREATE TABLE IF NOT EXISTS orbits (
    id serial primary key,
    x numeric NOT NULL,
    z numeric NOT NULL,
    v numeric NOT NULL,
    alpha numeric NOT NULL,
    t numeric NOT NULL,
    l1_r numeric,
    l2_r numeric,
    l3_r numeric,
    l4_r numeric,
    l5_r numeric,
    l6_r numeric,
    l1_im numeric,
    l2_im numeric,
    l3_im numeric,
    l4_im numeric,
    l5_im numeric,
    l6_im numeric,
    ax numeric NOT NULL,
    ay numeric NOT NULL,
    az numeric NOT NULL,
    dist_primary numeric NOT NULL,
    dist_secondary numeric NOT NULL,
    dist_curve numeric NOT NULL,
    cj numeric NOT NULL,
    stable bool,
    stability_order integer
);
CREATE INDEX ON orbits(x);
CREATE INDEX ON orbits(z);
CREATE INDEX ON orbits(v);
CREATE INDEX ON orbits(t);
CREATE INDEX ON orbits(ax);
CREATE INDEX ON orbits(ay);
CREATE INDEX ON orbits(az);
CREATE INDEX ON orbits(dist_primary);
CREATE INDEX ON orbits(dist_secondary);
CREATE INDEX ON orbits(dist_curve);
CREATE INDEX ON orbits(cj);


CREATE TABLE IF NOT EXISTS libration_points (
    id serial primary key,
    name varchar(40) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS orbit_families (
    id serial primary key,
    orbit_id int NOT NULL,
    tag varchar(20) NOT NULL,
    UNIQUE (orbit_id, tag)
);

CREATE TABLE IF NOT EXISTS trajectories (
    orbit_id int primary key,
    v jsonb NOT NULL,
    x numeric NOT NULL,
    y numeric NOT NULL,
    z numeric NOT NULL,
    vx numeric NOT NULL,
    vy numeric NOT NULL,
    vz numeric NOT NULL
);
CREATE INDEX ON trajectories(orbit_id);

CREATE TABLE IF NOT EXISTS poincare_sections (
    orbit_id int,
    plane varchar(8),
    v jsonb NOT NULL,

    PRIMARY KEY (orbit_id, plane)
);
CREATE INDEX ON poincare_sections(orbit_id, plane);