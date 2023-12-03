<script>
	import MainNav from './MainNav.svelte'
	import { siteAuthor } from '$lib/config'
</script>

<footer>
	<p>&copy;{new Date().getFullYear()} {siteAuthor}</p>
</footer>