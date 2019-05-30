import os
import jinja2
import settings
import pathlib

jobname = pathlib.Path.cwd().name


def write_submit(settings, jobname):

    # make jinja aware of templates
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath="./config")
    )

    template = jinja_env.get_template("submit.sh.jinja2")
    out = template.render(s=settings, jobname=jobname)

    fname = os.path.join("submit.sh")

    with open(fname, "w") as f:
        f.write(out)

    print(fname, "written.")

    return fname


if __name__ == "__main__":

    write_submit(settings, jobname)
