import os
import jinja2
import settings
import pathlib

jobname = pathlib.Path.cwd().name

templates = {"std":"submit.sh.jinja2", "array":"submit_array.sh.jinja2"}

template_file = templates[settings.template]

def write_submit(settings, jobname):

    # make jinja aware of templates
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath="./config")
    )

    template = jinja_env.get_template(template_file)
    out = template.render(s=settings, jobname=jobname)

    fname = os.path.join(template_file.rstrip("jinja2"))

    with open(fname, "w") as f:
        f.write(out)

    print(fname, "written.")

    return fname


if __name__ == "__main__":

    write_submit(settings, jobname)
